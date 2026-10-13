package api

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/modao/img/internal/ai"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type toolArguments struct {
	SessionID    string      `json:"sessionId" jsonschema:"Temporary image editing session ID"`
	BaseRevision string      `json:"baseRevision,omitempty" jsonschema:"Revision obtained by inspect_asset; required for proposal or edit changes"`
	ProposalID   string      `json:"proposalId,omitempty"`
	Accepted     bool        `json:"accepted,omitempty" jsonschema:"Must be explicitly true to apply or undo operations"`
	Operations   []Operation `json:"operations,omitempty"`
	Summary      string      `json:"summary,omitempty"`
	EditGraph    *EditGraph  `json:"editGraph,omitempty"`
	Format       string      `json:"format,omitempty"`
	Quality      int         `json:"quality,omitempty"`
}

type toolInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	ReadOnly    bool   `json:"readOnly"`
	Available   bool   `json:"available"`
	Runtime     string `json:"runtime"`
}

func toolCatalog() []toolInfo {
	return []toolInfo{
		{"inspect_asset", "Inspect current image, graph revision and supported operation parameter catalog", true, true, "go"},
		{"propose_operations", "Store a non-destructive edit proposal from Operation parameters. Requires current baseRevision. This tool does not run a language model.", false, true, "go"},
		{"apply_operations", "Append an existing proposal only with accepted=true and matching baseRevision", false, true, "go"},
		{"render_preview", "Render the current graph through the Rust image core and return an image", true, true, "rust-native"},
		{"export_image", "Create a cancellable Rust export job and return its result URL", false, true, "rust-native"},
		{"undo_operations", "Restore previous graph as a new revision. Requires accepted=true and current baseRevision", false, true, "go"},
		{"detect_landmarks", "Runs in the browser MediaPipe provider; not a server inference tool", true, false, "browser-mediapipe"},
		{"detect_regions", "Runs in the browser segmentation provider; not a server inference tool", true, false, "browser-mediapipe"},
	}
}

//go:embed operation-catalog.json
var operationCatalogJSON []byte

func operationCatalog() map[string]any {
	var catalog map[string]any
	_ = json.Unmarshal(operationCatalogJSON, &catalog)
	return catalog
}

func (s *Server) MCPServer() *mcp.Server {
	server := mcp.NewServer(&mcp.Implementation{Name: "img-editor", Version: "0.2.0"}, nil)
	for _, info := range toolCatalog() {
		name := info.Name
		destructive := false
		openWorld := false
		tool := &mcp.Tool{Name: name, Description: info.Description, Annotations: &mcp.ToolAnnotations{ReadOnlyHint: info.ReadOnly, DestructiveHint: &destructive, OpenWorldHint: &openWorld}}
		mcp.AddTool[toolArguments, any](server, tool, func(ctx context.Context, _ *mcp.CallToolRequest, args toolArguments) (*mcp.CallToolResult, any, error) {
			result, output := s.executeTool(ctx, ai.ToolName(name), args)
			response := &mcp.CallToolResult{IsError: result.Status == ai.ToolResultStatusError}
			encoded, _ := json.Marshal(result)
			response.Content = []mcp.Content{&mcp.TextContent{Text: string(encoded)}}
			if output != nil {
				response.Content = append(response.Content, &mcp.ImageContent{Data: output.Data, MIMEType: "image/" + output.Metadata.Format})
			}
			return response, result, nil
		})
	}
	return server
}

func (s *Server) MCPHandler() http.Handler {
	server := s.MCPServer()
	return mcp.NewStreamableHTTPHandler(func(_ *http.Request) *mcp.Server { return server }, &mcp.StreamableHTTPOptions{JSONResponse: true, SessionTimeout: 30 * time.Minute, MaxRequestBodyBytes: 8 << 20})
}

func toolFailure(id string, name ai.ToolName, code, message string) ai.ToolResult {
	return ai.ToolResult{ID: id, Name: name, Status: ai.ToolResultStatusError, Error: &ai.ToolError{Code: code, Message: message}}
}
func (s *Server) executeTool(ctx context.Context, name ai.ToolName, args toolArguments) (ai.ToolResult, *renderOutput) {
	id, _ := newID()
	result := ai.ToolResult{ID: id, Name: name, Status: ai.ToolResultStatusOK}
	sess, ok := s.getSession(args.SessionID)
	if !ok || sess.Image == nil {
		return toolFailure(id, name, "session_not_found", "upload an image to the session first"), nil
	}
	switch name {
	case ai.ToolInspectAsset:
		result.Result = map[string]any{"asset": sess.Image, "revision": sess.Revision, "editGraph": sess.EditGraph, "operations": operationCatalog(), "tools": toolCatalog(), "coordinateSystem": "region/landmark coordinates normalized 0..1; crop/resize use pixels"}
	case ai.ToolProposeOperations:
		proposal, err := s.propose(ctx, args.SessionID, args.BaseRevision, args.Operations, args.Summary)
		if err != nil {
			return toolFailure(id, name, "proposal_rejected", err.Error()), nil
		}
		result.Status = ai.ToolResultStatusPendingApproval
		result.Proposal = &proposal
	case ai.ToolApplyOperations:
		graph, revision, err := s.applyProposal(ctx, args.SessionID, args.ProposalID, args.BaseRevision, args.Accepted)
		if err != nil {
			return toolFailure(id, name, "apply_rejected", err.Error()), nil
		}
		result.Result = map[string]any{"editGraph": graph, "revision": revision}
	case ai.ToolUndoOperations:
		if !args.Accepted {
			return toolFailure(id, name, "approval_required", "undo requires explicit accepted=true"), nil
		}
		graph, revision, err := s.undo(args.SessionID, args.BaseRevision)
		if err != nil {
			return toolFailure(id, name, "undo_rejected", err.Error()), nil
		}
		result.Result = map[string]any{"editGraph": graph, "revision": revision}
	case ai.ToolRenderPreview:
		output, err := s.runRender(ctx, args.SessionID, renderRequest{EditGraph: args.EditGraph, Format: args.Format, Quality: args.Quality}, nil)
		if err != nil {
			return toolFailure(id, name, "render_failed", publicToolRenderError(err)), nil
		}
		result.Result = map[string]any{"width": output.Metadata.Width, "height": output.Metadata.Height, "format": output.Metadata.Format, "revision": sess.Revision}
		return result, &output
	case ai.ToolExportImage:
		job, err := s.startJob(args.SessionID, renderRequest{EditGraph: args.EditGraph, Format: args.Format, Quality: args.Quality})
		if err != nil {
			return toolFailure(id, name, "export_rejected", publicToolRenderError(err)), nil
		}
		result.Result = map[string]any{"job": job, "statusUrl": "/api/v1/jobs/" + job.ID}
	case ai.ToolDetectLandmarks, ai.ToolDetectRegions:
		return toolFailure(id, name, "browser_provider_required", "use the browser MediaPipe provider; this server does not return fabricated landmarks or masks"), nil
	default:
		return toolFailure(id, name, "unknown_tool", "unsupported tool name"), nil
	}
	return result, nil
}

func publicToolRenderError(err error) string {
	if errors.Is(err, errWorkerRejected) || errors.Is(err, errWorkerUnavailable) {
		return err.Error()
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "render timed out"
	}
	if errors.Is(err, context.Canceled) {
		return "render cancelled"
	}
	return "render failed"
}

func (s *Server) propose(ctx context.Context, sessionID, baseRevision string, operations []Operation, summary string) (ai.EditProposal, error) {
	if len(operations) == 0 || len(operations) > 1024 {
		return ai.EditProposal{}, errors.New("a proposal requires 1..1024 operations")
	}
	snapshot, ok := s.getSession(sessionID)
	if !ok || snapshot.Image == nil {
		return ai.EditProposal{}, errors.New("session image not found")
	}
	if baseRevision == "" || baseRevision != snapshot.Revision {
		return ai.EditProposal{}, errors.New("baseRevision does not match the current session revision")
	}
	operations = cloneGraph(EditGraph{Operations: operations}).Operations
	for index, op := range operations {
		op = normalizeOperation(op)
		op.Origin = "ai"
		if op.Provenance == nil {
			op.Provenance = &ai.Provenance{ActorType: "ai", ToolName: "propose_operations"}
		}
		if err := op.Validate(); err != nil {
			return ai.EditProposal{}, err
		}
		operations[index] = op
	}
	graph := cloneGraph(snapshot.EditGraph)
	graph.Operations = append(graph.Operations, operations...)
	if _, err := s.runRender(ctx, sessionID, renderRequest{EditGraph: &graph, Format: "png", forceCore: true}, nil); err != nil {
		return ai.EditProposal{}, errors.New(publicToolRenderError(err))
	}
	id, err := newID()
	if err != nil {
		return ai.EditProposal{}, errors.New("proposal id generation failed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[sessionID]
	if !ok || !s.now().Before(sess.ExpiresAt) {
		return ai.EditProposal{}, errors.New("session expired")
	}
	if baseRevision == "" || baseRevision != sess.Revision {
		return ai.EditProposal{}, errors.New("baseRevision does not match the current session revision")
	}
	if len(sess.proposals) >= 256 {
		return ai.EditProposal{}, errors.New("session proposal limit reached")
	}
	proposal := ai.EditProposal{ID: id, BaseRevision: baseRevision, Operations: operations, Status: ai.ProposalStatusProposed, CreatedBy: ai.ProposalActor{ActorType: "ai"}, ApprovalMode: ai.ApprovalModeConfirm, Summary: summary}
	sess.proposals[id] = proposal
	return proposal, nil
}

func (s *Server) applyProposal(ctx context.Context, sessionID, proposalID, baseRevision string, accepted bool) (EditGraph, string, error) {
	if !accepted {
		return EditGraph{}, "", errors.New("applying a proposal requires explicit accepted=true")
	}
	s.mu.Lock()
	sess, ok := s.sessions[sessionID]
	if !ok || !s.now().Before(sess.ExpiresAt) {
		s.mu.Unlock()
		return EditGraph{}, "", errors.New("session expired")
	}
	proposal, ok := sess.proposals[proposalID]
	if !ok {
		s.mu.Unlock()
		return EditGraph{}, "", errors.New("proposal not found")
	}
	if proposal.Status != ai.ProposalStatusProposed {
		s.mu.Unlock()
		return EditGraph{}, "", fmt.Errorf("proposal cannot be applied in status %s", proposal.Status)
	}
	if baseRevision == "" || baseRevision != sess.Revision || proposal.BaseRevision != sess.Revision {
		proposal.Status = ai.ProposalStatusExpired
		sess.proposals[proposalID] = proposal
		s.mu.Unlock()
		return EditGraph{}, "", errors.New("revision conflict; create a new proposal")
	}
	graph := cloneGraph(sess.EditGraph)
	graph.Operations = append(graph.Operations, proposal.Operations...)
	s.mu.Unlock()
	// Validate the exact combined graph through the shared Rust engine before
	// committing. No invalid parameter can silently become an applied edit.
	if _, err := s.runRender(ctx, sessionID, renderRequest{EditGraph: &graph, Format: "png", forceCore: true}, nil); err != nil {
		return EditGraph{}, "", errors.New(publicToolRenderError(err))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok = s.sessions[sessionID]
	if !ok || !s.now().Before(sess.ExpiresAt) || sess.Revision != baseRevision {
		return EditGraph{}, "", errors.New("revision changed while rendering; retry with a new proposal")
	}
	currentProposal, ok := sess.proposals[proposalID]
	if !ok || currentProposal.Status != ai.ProposalStatusProposed {
		return EditGraph{}, "", errors.New("proposal changed while rendering")
	}
	sess.history = append(sess.history, cloneGraph(sess.EditGraph))
	sess.EditGraph = graph
	sess.revisionNumber++
	sess.Revision = fmt.Sprint(sess.revisionNumber)
	proposal.Status = ai.ProposalStatusApplied
	sess.proposals[proposalID] = proposal
	return cloneGraph(graph), sess.Revision, nil
}

func (s *Server) rejectProposal(sessionID, proposalID, baseRevision string) (ai.EditProposal, EditGraph, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[sessionID]
	if !ok || !s.now().Before(sess.ExpiresAt) {
		return ai.EditProposal{}, EditGraph{}, "", errors.New("session expired")
	}
	if baseRevision == "" || baseRevision != sess.Revision {
		return ai.EditProposal{}, EditGraph{}, "", errors.New("revision conflict")
	}
	proposal, ok := sess.proposals[proposalID]
	if !ok {
		return ai.EditProposal{}, EditGraph{}, "", errors.New("proposal not found")
	}
	if proposal.Status != ai.ProposalStatusProposed {
		return ai.EditProposal{}, EditGraph{}, "", fmt.Errorf("proposal cannot be rejected in status %s", proposal.Status)
	}
	proposal.Status = ai.ProposalStatusRejected
	sess.proposals[proposalID] = proposal
	return proposal, cloneGraph(sess.EditGraph), sess.Revision, nil
}

func (s *Server) appendManual(ctx context.Context, sessionID, baseRevision string, operations []Operation) (EditGraph, string, error) {
	if len(operations) == 0 {
		return EditGraph{}, "", errors.New("operations cannot be empty")
	}
	sess, ok := s.getSession(sessionID)
	if !ok || sess.Image == nil {
		return EditGraph{}, "", errors.New("session image not found")
	}
	if baseRevision == "" || sess.Revision != baseRevision {
		return EditGraph{}, "", errors.New("revision conflict")
	}
	graph := cloneGraph(sess.EditGraph)
	for _, op := range operations {
		op = normalizeOperation(op)
		op.Origin = "manual"
		if err := op.Validate(); err != nil {
			return EditGraph{}, "", err
		}
		graph.Operations = append(graph.Operations, op)
	}
	if _, err := s.runRender(ctx, sessionID, renderRequest{EditGraph: &graph, Format: "png", forceCore: true}, nil); err != nil {
		return EditGraph{}, "", errors.New(publicToolRenderError(err))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	current, ok := s.sessions[sessionID]
	if !ok || current.Revision != baseRevision {
		return EditGraph{}, "", errors.New("revision conflict")
	}
	current.history = append(current.history, cloneGraph(current.EditGraph))
	current.EditGraph = graph
	current.revisionNumber++
	current.Revision = fmt.Sprint(current.revisionNumber)
	return cloneGraph(graph), current.Revision, nil
}

func (s *Server) undo(sessionID, baseRevision string) (EditGraph, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[sessionID]
	if !ok || !s.now().Before(sess.ExpiresAt) {
		return EditGraph{}, "", errors.New("session expired")
	}
	if baseRevision == "" || sess.Revision != baseRevision {
		return EditGraph{}, "", errors.New("revision conflict")
	}
	if len(sess.history) == 0 {
		return EditGraph{}, "", errors.New("no previous revision")
	}
	sess.EditGraph = sess.history[len(sess.history)-1]
	sess.history = sess.history[:len(sess.history)-1]
	sess.revisionNumber++
	sess.Revision = fmt.Sprint(sess.revisionNumber)
	return cloneGraph(sess.EditGraph), sess.Revision, nil
}

func (s *Server) listProposalsHTTP(c *gin.Context) {
	s.mu.Lock()
	sess, ok := s.sessions[c.Param("id")]
	if !ok || !s.now().Before(sess.ExpiresAt) {
		s.mu.Unlock()
		notFound(c.Writer)
		return
	}
	proposals := []ai.EditProposal{}
	for _, p := range sess.proposals {
		proposals = append(proposals, p)
	}
	data, _ := json.Marshal(map[string]any{"proposals": proposals})
	s.mu.Unlock()
	c.Data(200, "application/json", data)
}
func decodeToolArgs(c *gin.Context) (toolArguments, bool) {
	var args toolArguments
	if json.NewDecoder(http.MaxBytesReader(c.Writer, c.Request.Body, 8<<20)).Decode(&args) != nil {
		writeError(c.Writer, 400, "invalid_arguments", "expected tool arguments JSON")
		return args, false
	}
	args.SessionID = c.Param("id")
	return args, true
}
func (s *Server) proposeHTTP(c *gin.Context) {
	args, ok := decodeToolArgs(c)
	if !ok {
		return
	}
	p, err := s.propose(c.Request.Context(), args.SessionID, args.BaseRevision, args.Operations, args.Summary)
	if err != nil {
		writeError(c.Writer, 409, "proposal_rejected", err.Error())
		return
	}
	c.JSON(201, p)
}
func (s *Server) applyHTTP(c *gin.Context) {
	args, ok := decodeToolArgs(c)
	if !ok {
		return
	}
	graph, revision, err := s.applyProposal(c.Request.Context(), args.SessionID, c.Param("proposalID"), args.BaseRevision, args.Accepted)
	if err != nil {
		writeError(c.Writer, 409, "apply_rejected", err.Error())
		return
	}
	c.JSON(200, gin.H{"revision": revision, "editGraph": graph})
}
func (s *Server) rejectHTTP(c *gin.Context) {
	args, ok := decodeToolArgs(c)
	if !ok {
		return
	}
	proposal, graph, revision, err := s.rejectProposal(args.SessionID, c.Param("proposalID"), args.BaseRevision)
	if err != nil {
		writeError(c.Writer, 409, "reject_rejected", err.Error())
		return
	}
	c.JSON(200, gin.H{"proposal": proposal, "revision": revision, "editGraph": graph})
}
func (s *Server) operationsHTTP(c *gin.Context) {
	args, ok := decodeToolArgs(c)
	if !ok {
		return
	}
	graph, revision, err := s.appendManual(c.Request.Context(), args.SessionID, args.BaseRevision, args.Operations)
	if err != nil {
		writeError(c.Writer, 409, "operation_rejected", err.Error())
		return
	}
	c.JSON(200, gin.H{"revision": revision, "editGraph": graph})
}
func (s *Server) undoHTTP(c *gin.Context) {
	args, ok := decodeToolArgs(c)
	if !ok {
		return
	}
	graph, revision, err := s.undo(args.SessionID, args.BaseRevision)
	if err != nil {
		writeError(c.Writer, 409, "undo_rejected", err.Error())
		return
	}
	c.JSON(200, gin.H{"revision": revision, "editGraph": graph})
}
func (s *Server) toolHTTP(c *gin.Context) {
	var args toolArguments
	if json.NewDecoder(http.MaxBytesReader(c.Writer, c.Request.Body, 8<<20)).Decode(&args) != nil {
		writeError(c.Writer, 400, "invalid_arguments", "expected tool arguments JSON")
		return
	}
	result, _ := s.executeTool(c.Request.Context(), ai.ToolName(strings.TrimSpace(c.Param("name"))), args)
	c.JSON(200, result)
}

// replaceGraph makes an explicitly saved manual graph visible to MCP without
// forcing browser slider events to use the server for every frame.
func (s *Server) replaceGraph(ctx context.Context, sessionID, baseRevision string, graph *EditGraph) (EditGraph, string, error) {
	if graph == nil {
		return EditGraph{}, "", errors.New("editGraph is required")
	}
	sess, ok := s.getSession(sessionID)
	if !ok || sess.Image == nil {
		return EditGraph{}, "", errors.New("session image not found")
	}
	if baseRevision == "" || sess.Revision != baseRevision {
		return EditGraph{}, "", errors.New("revision conflict")
	}
	prepared, err := prepareGraph(renderRequest{EditGraph: graph}, sess)
	if err != nil {
		return EditGraph{}, "", err
	}
	if _, err := s.runRender(ctx, sessionID, renderRequest{EditGraph: &prepared, Format: "png", forceCore: true}, nil); err != nil {
		return EditGraph{}, "", errors.New(publicToolRenderError(err))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	current, ok := s.sessions[sessionID]
	if !ok || !s.now().Before(current.ExpiresAt) || current.Revision != baseRevision {
		return EditGraph{}, "", errors.New("revision conflict")
	}
	current.history = append(current.history, cloneGraph(current.EditGraph))
	current.EditGraph = prepared
	current.revisionNumber++
	current.Revision = fmt.Sprint(current.revisionNumber)
	return cloneGraph(prepared), current.Revision, nil
}
func (s *Server) graphHTTP(c *gin.Context) {
	args, ok := decodeToolArgs(c)
	if !ok {
		return
	}
	graph, revision, err := s.replaceGraph(c.Request.Context(), args.SessionID, args.BaseRevision, args.EditGraph)
	if err != nil {
		writeError(c.Writer, 409, "graph_rejected", err.Error())
		return
	}
	c.JSON(200, gin.H{"revision": revision, "editGraph": graph})
}
