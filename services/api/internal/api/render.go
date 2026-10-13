package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

var errWorkerRejected = errors.New("image-core rejected the editing parameters")
var errWorkerUnavailable = errors.New("Rust image-render worker is unavailable; build core/image-core first")

// findRenderBinary resolves the repository binary from either the API working
// directory, the repo working directory, or a packaged executable's directory.
func findRenderBinary() string {
	candidates := []string{}
	if cwd, err := os.Getwd(); err == nil {
		for dir := cwd; ; dir = filepath.Dir(dir) {
			candidates = append(candidates, filepath.Join(dir, "core/image-core/target/release/image-render"))
			if filepath.Dir(dir) == dir {
				break
			}
		}
	}
	if executable, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(executable), "image-render"))
	}
	_, source, _, _ := runtime.Caller(0)
	candidates = append(candidates, filepath.Join(filepath.Dir(source), "../../../../core/image-core/target/release/image-render"))
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			absolute, _ := filepath.Abs(candidate)
			return absolute
		}
	}
	return filepath.Clean(candidates[len(candidates)-1])
}

type renderMetadata struct {
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Format string `json:"format"`
}

type renderOutput struct {
	Data     []byte
	Metadata renderMetadata
}

func decodeRenderRequest(r *http.Request) (renderRequest, error) {
	var req renderRequest
	if r.Body == nil || r.ContentLength == 0 {
		return req, nil
	}
	dec := json.NewDecoder(io.LimitReader(r.Body, (8<<20)+1))
	if err := dec.Decode(&req); err != nil {
		return req, errors.New("expected JSON render request")
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		return req, errors.New("render request must contain one JSON object")
	}
	return req, nil
}

func normalizeOperation(op Operation) Operation {
	if op.ID == "" {
		op.ID, _ = newID()
	}
	if op.Version == "" {
		op.Version = "1.0"
	}
	if op.Params == nil {
		op.Params = map[string]any{}
	}
	return op
}

func prepareGraph(req renderRequest, sess *session) (EditGraph, error) {
	graph := cloneGraph(sess.EditGraph)
	if req.EditGraph != nil {
		graph = cloneGraph(*req.EditGraph)
	}
	if graph.SourceAssetID != "" && graph.SourceAssetID != sess.Image.ID {
		return graph, errors.New("sourceAssetId must match the uploaded session asset")
	}
	graph.SourceAssetID = sess.Image.ID
	if graph.Version == "" {
		graph.Version = "1.0"
	}
	if len(graph.Nodes) > 0 {
		graph.Operations = append(graph.Operations, graph.Nodes...)
		graph.Nodes = nil
	}
	if req.Operation != nil {
		graph.Operations = append(graph.Operations, *req.Operation)
	}
	if len(graph.Operations) > 1024 {
		return graph, errors.New("too many operations")
	}
	for i, op := range graph.Operations {
		graph.Operations[i] = normalizeOperation(op)
		if err := graph.Operations[i].Validate(); err != nil {
			return graph, err
		}
	}
	return graph, nil
}

func renderFormat(req renderRequest, a *asset) (string, int, error) {
	format := strings.ToLower(req.Format)
	if format == "" || format == "auto" {
		format = strings.TrimPrefix(a.MimeType, "image/")
		if format != "png" && format != "jpeg" {
			format = "png"
		}
	}
	if format == "jpg" {
		format = "jpeg"
	}
	if format != "png" && format != "jpeg" {
		return "", 0, errors.New("format must be png or jpeg")
	}
	quality := req.Quality
	if quality == 0 {
		quality = 92
	}
	if quality < 1 || quality > 100 {
		return "", 0, errors.New("quality must be between 1 and 100")
	}
	return format, quality, nil
}

// Render never executes pixel loops in Go. Only an empty graph can bypass the
// Rust core; even disabled and identity nodes require native parameter checks.
func (s *Server) runRender(ctx context.Context, sessionID string, req renderRequest, progress func(float64, string)) (renderOutput, error) {
	sess, ok := s.getSession(sessionID)
	if !ok || sess.Image == nil {
		return renderOutput{}, errors.New("session image not found")
	}
	graph, err := prepareGraph(req, sess)
	if err != nil {
		return renderOutput{}, fmt.Errorf("%w: %s", errWorkerRejected, err)
	}
	format, quality, err := renderFormat(req, sess.Image)
	if err != nil {
		return renderOutput{}, fmt.Errorf("%w: %s", errWorkerRejected, err)
	}
	ctx, cancel := context.WithTimeout(ctx, s.renderTimeout)
	defer cancel()
	select {
	case s.renderSlots <- struct{}{}:
		defer func() { <-s.renderSlots }()
	case <-ctx.Done():
		return renderOutput{}, ctx.Err()
	}
	if progress != nil {
		progress(0.1, "running")
	}
	// Open while holding the session lock; replacement/deletion cannot invalidate
	// a source before the worker's request-local snapshot has an open descriptor.
	s.mu.RLock()
	current, exists := s.sessions[sessionID]
	var input *os.File
	if exists && current.Image != nil && current.Image.ID == sess.Image.ID {
		input, err = os.Open(sess.Image.Path)
	} else {
		err = errors.New("source asset changed")
	}
	s.mu.RUnlock()
	if err != nil {
		return renderOutput{}, errors.New("session image changed or expired")
	}
	defer input.Close()
	allDisabled := len(graph.Operations) > 0
	for _, op := range graph.Operations {
		if op.Enabled {
			allDisabled = false
		}
	}
	if !req.forceCore && len(graph.Operations) == 0 && sess.Image.MimeType == "image/"+format {
		data, err := io.ReadAll(input)
		return renderOutput{Data: data, Metadata: renderMetadata{Width: sess.Image.Width, Height: sess.Image.Height, Format: format}}, err
	}
	if _, err := os.Stat(s.renderBinary); err != nil {
		return renderOutput{}, errWorkerUnavailable
	}
	dir, err := os.MkdirTemp(s.dataDir, "render-*")
	if err != nil {
		return renderOutput{}, errors.New("render temporary storage is unavailable")
	}
	defer os.RemoveAll(dir)
	inputPath := filepath.Join(dir, "input")
	outputPath := filepath.Join(dir, "output."+format)
	graphPath := filepath.Join(dir, "graph.json")
	snapshot, err := os.Create(inputPath)
	if err != nil {
		return renderOutput{}, errors.New("render temporary storage is unavailable")
	}
	_, copyErr := io.Copy(snapshot, input)
	closeErr := snapshot.Close()
	if copyErr != nil || closeErr != nil {
		return renderOutput{}, errors.New("source snapshot failed")
	}
	data, err := json.Marshal(graph)
	if err != nil {
		return renderOutput{}, errWorkerRejected
	}
	if err := os.WriteFile(graphPath, data, 0o600); err != nil {
		return renderOutput{}, errors.New("render graph storage failed")
	}
	cmd := exec.CommandContext(ctx, s.renderBinary, "--input", inputPath, "--output", outputPath, "--graph", graphPath, "--format", format, "--quality", fmt.Sprint(quality))
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return renderOutput{}, ctx.Err()
		}
		if _, ok := err.(*exec.ExitError); ok {
			return renderOutput{}, fmt.Errorf("%w: %s", errWorkerRejected, safeWorkerMessage(stderr.String(), dir, sess.Image.Path))
		}
		return renderOutput{}, errWorkerUnavailable
	}
	metadata := renderMetadata{}
	if json.Unmarshal(stdout.Bytes(), &metadata) != nil || metadata.Width <= 0 || metadata.Height <= 0 || metadata.Format != format {
		return renderOutput{}, errors.New("invalid render worker response")
	}
	if progress != nil {
		progress(0.9, "encoding")
	}
	result, err := os.ReadFile(outputPath)
	if err != nil {
		return renderOutput{}, errors.New("render output is unavailable")
	}
	if allDisabled && sess.Image.MimeType == "image/"+format {
		// Native validation has already checked every disabled node. Preserve
		// the source encoding for this exact no-op; enabled identity nodes may
		// still alter pixels through common compositing and must keep output.
		result, err = os.ReadFile(inputPath)
		if err != nil {
			return renderOutput{}, errors.New("source snapshot is unavailable")
		}
	}
	return renderOutput{Data: result, Metadata: metadata}, nil
}

func safeWorkerMessage(message string, paths ...string) string {
	message = strings.TrimSpace(message)
	for _, path := range paths {
		if path != "" {
			message = strings.ReplaceAll(message, path, "[temporary asset]")
		}
	}
	if strings.Contains(message, "/") || strings.Contains(message, "\\") {
		return "invalid operation parameters or unsupported image"
	}
	if len(message) > 512 {
		message = message[:512]
	}
	if message == "" {
		message = "invalid operation parameters"
	}
	return message
}

func (s *Server) render(w http.ResponseWriter, r *http.Request, sessionID string) {
	req, err := decodeRenderRequest(r)
	if err != nil {
		writeError(w, 400, "invalid_render_request", err.Error())
		return
	}
	output, err := s.runRender(r.Context(), sessionID, req, nil)
	if err != nil {
		writeRenderError(w, err)
		return
	}
	w.Header().Set("Content-Type", "image/"+output.Metadata.Format)
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Image-Width", fmt.Sprint(output.Metadata.Width))
	w.Header().Set("X-Image-Height", fmt.Sprint(output.Metadata.Height))
	_, _ = w.Write(output.Data)
}

func writeRenderError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, errWorkerRejected):
		writeError(w, 422, "invalid_operation", err.Error())
	case errors.Is(err, errWorkerUnavailable):
		writeError(w, 503, "render_worker_unavailable", err.Error())
	case errors.Is(err, context.DeadlineExceeded):
		writeError(w, 504, "render_timeout", "render exceeded the configured time limit")
	case errors.Is(err, context.Canceled):
		writeError(w, 409, "render_cancelled", "render was cancelled")
	default:
		writeError(w, 500, "render_failed", "render could not be completed")
	}
}
