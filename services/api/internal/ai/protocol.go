// Package ai contains the transport-neutral protocol shared by the Go API,
// the DeepAgent gateway and MCP adapters. It describes proposals and tool
// messages only; it deliberately does not execute tools or render images.
package ai

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ApprovalMode controls whether an AI generated action may be applied to the
// current edit graph. The default policy at the API boundary is confirm.
type ApprovalMode string

const (
	ApprovalModeAuto     ApprovalMode = "auto"
	ApprovalModeConfirm  ApprovalMode = "confirm"
	ApprovalModeDisabled ApprovalMode = "disabled"
)

// IsValid reports whether mode is a protocol value.
func (mode ApprovalMode) IsValid() bool {
	switch mode {
	case ApprovalModeAuto, ApprovalModeConfirm, ApprovalModeDisabled:
		return true
	default:
		return false
	}
}

// ToolName is the allow-listed name exposed to an AI/MCP client. Keeping this
// list finite lets the API reject arbitrary function names before dispatch.
type ToolName string

const (
	ToolInspectAsset      ToolName = "inspect_asset"
	ToolDetectLandmarks   ToolName = "detect_landmarks"
	ToolDetectRegions     ToolName = "detect_regions"
	ToolProposeOperations ToolName = "propose_operations"
	ToolApplyOperations   ToolName = "apply_operations"
	ToolRenderPreview     ToolName = "render_preview"
	ToolExportImage       ToolName = "export_image"
	ToolUndoOperations    ToolName = "undo_operations"
)

// IsValid reports whether name is a supported tool name.
func (name ToolName) IsValid() bool {
	switch name {
	case ToolInspectAsset, ToolDetectLandmarks, ToolDetectRegions,
		ToolProposeOperations, ToolApplyOperations, ToolRenderPreview,
		ToolExportImage, ToolUndoOperations:
		return true
	default:
		return false
	}
}

// ProposalStatus is the lifecycle of an AI edit proposal.
type ProposalStatus string

const (
	ProposalStatusProposed ProposalStatus = "proposed"
	ProposalStatusAccepted ProposalStatus = "accepted"
	ProposalStatusRejected ProposalStatus = "rejected"
	ProposalStatusApplied  ProposalStatus = "applied"
	ProposalStatusExpired  ProposalStatus = "expired"
)

func (status ProposalStatus) IsValid() bool {
	switch status {
	case ProposalStatusProposed, ProposalStatusAccepted, ProposalStatusRejected,
		ProposalStatusApplied, ProposalStatusExpired:
		return true
	default:
		return false
	}
}

// ToolResultStatus is the outcome of a tool call. pending_approval is a
// successful protocol response that tells the client to obtain user consent.
type ToolResultStatus string

const (
	ToolResultStatusOK              ToolResultStatus = "ok"
	ToolResultStatusError           ToolResultStatus = "error"
	ToolResultStatusPendingApproval ToolResultStatus = "pending_approval"
	ToolResultStatusRejected        ToolResultStatus = "rejected"
)

func (status ToolResultStatus) IsValid() bool {
	switch status {
	case ToolResultStatusOK, ToolResultStatusError,
		ToolResultStatusPendingApproval, ToolResultStatusRejected:
		return true
	default:
		return false
	}
}

// ControlPoint mirrors schemas/operation.schema.json. Coordinates are in the
// source image coordinate space and optional handles are used by spline tools.
type ControlPoint struct {
	X    float64 `json:"x"`
	Y    float64 `json:"y"`
	InX  float64 `json:"inX,omitempty"`
	InY  float64 `json:"inY,omitempty"`
	OutX float64 `json:"outX,omitempty"`
	OutY float64 `json:"outY,omitempty"`
}

// Provenance records who or what produced an operation.
type Provenance struct {
	ActorType  string  `json:"actorType"`
	ActorID    string  `json:"actorId,omitempty"`
	ToolName   string  `json:"toolName,omitempty"`
	Model      string  `json:"model,omitempty"`
	RequestID  string  `json:"requestId,omitempty"`
	Confidence float64 `json:"confidence,omitempty"`
}

// Operation is the AI protocol's copy of the shared non-destructive edit
// operation. It intentionally uses map[string]any for operation-specific
// parameters so new image tools do not require a Go release.
type Operation struct {
	ID            string         `json:"id"`
	Type          string         `json:"type"`
	Version       string         `json:"version"`
	Enabled       bool           `json:"enabled"`
	Origin        string         `json:"origin,omitempty"`
	Label         string         `json:"label,omitempty"`
	Explanation   string         `json:"explanation,omitempty"`
	Provenance    *Provenance    `json:"provenance,omitempty"`
	MaskID        *string        `json:"maskId,omitempty"`
	TargetID      *string        `json:"targetId,omitempty"`
	Params        map[string]any `json:"params"`
	ControlPoints []ControlPoint `json:"controlPoints,omitempty"`
	EngineVersion string         `json:"engineVersion,omitempty"`
}

// UnmarshalJSON keeps legacy demo requests compatible: an omitted enabled
// field means enabled, while an explicit false always remains disabled.
func (operation *Operation) UnmarshalJSON(data []byte) error {
	type plain Operation
	decoded := plain{Enabled: true}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*operation = Operation(decoded)
	return nil
}

// Validate checks the operation fields that must be safe before a proposal
// can cross the API boundary. Operation-specific ranges remain owned by the
// corresponding tool schema and render engine.
func (operation Operation) Validate() error {
	if operation.ID == "" {
		return errors.New("operation id is required")
	}
	if operation.Version == "" {
		return errors.New("operation version is required")
	}
	if !isProtocolVersion(operation.Version) {
		return fmt.Errorf("invalid operation version %q", operation.Version)
	}
	if operation.Type == "" || operation.Type != strings.TrimSpace(operation.Type) || !isOperationType(operation.Type) {
		return fmt.Errorf("unsupported operation type %q", operation.Type)
	}
	if operation.Params == nil {
		return errors.New("operation params are required")
	}
	if operation.Origin != "" && operation.Origin != "manual" && operation.Origin != "ai" && operation.Origin != "imported" {
		return fmt.Errorf("unsupported operation origin %q", operation.Origin)
	}
	return nil
}

func isProtocolVersion(version string) bool {
	parts := strings.Split(version, ".")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return false
	}
	for _, part := range parts {
		for _, character := range part {
			if character < '0' || character > '9' {
				return false
			}
		}
	}
	return true
}

func isOperationType(operationType string) bool {
	switch strings.TrimSpace(operationType) {
	case "identity", "crop", "resize", "rotate", "flip", "perspective", "warp.grid", "warp.liquify",
		"face.adjust", "feature.adjust", "skin.retouch", "body.adjust", "clothing.adjust",
		"repair.clone", "repair.heal", "repair.inpaint", "paint.brush", "color.basic", "color.curves", "color.hsl", "filter.film", "filter.lut":
		return true
	default:
		return false
	}
}

// ProposalActor identifies the actor that created an edit proposal.
type ProposalActor struct {
	ActorType  string `json:"actorType"`
	ActorID    string `json:"actorId,omitempty"`
	Model      string `json:"model,omitempty"`
	ToolCallID string `json:"toolCallId,omitempty"`
}

// EditProposal contains operations that can be reviewed and appended to an
// EditGraph. A proposal never contains replacement source pixels.
type EditProposal struct {
	ID           string         `json:"id"`
	BaseRevision string         `json:"baseRevision"`
	Operations   []Operation    `json:"operations"`
	Status       ProposalStatus `json:"status"`
	CreatedBy    ProposalActor  `json:"createdBy"`
	ApprovalMode ApprovalMode   `json:"approvalMode,omitempty"`
	Summary      string         `json:"summary,omitempty"`
	Warnings     []string       `json:"warnings,omitempty"`
}

// ToolCall is the JSON envelope sent to an AI/MCP tool adapter. Arguments are
// deliberately opaque to this package; each tool owns its argument schema.
type ToolCall struct {
	ID           string         `json:"id,omitempty"`
	Name         ToolName       `json:"name"`
	Arguments    map[string]any `json:"arguments"`
	ApprovalMode ApprovalMode   `json:"approvalMode,omitempty"`
	ProposalID   string         `json:"proposalId,omitempty"`
	SessionID    string         `json:"sessionId,omitempty"`
	RequestID    string         `json:"requestId,omitempty"`
}

// ToolError is safe to expose to a client and does not include internal stack
// traces or image paths.
type ToolError struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable,omitempty"`
}

// ToolResult is the response envelope correlated by ToolCall.ID. Result is a
// JSON object so tool-specific output can evolve without changing this wire
// protocol. A proposal is returned separately to make approval explicit.
type ToolResult struct {
	ID       string           `json:"id"`
	Status   ToolResultStatus `json:"status"`
	Name     ToolName         `json:"name,omitempty"`
	Result   map[string]any   `json:"result,omitempty"`
	Proposal *EditProposal    `json:"proposal,omitempty"`
	Error    *ToolError       `json:"error,omitempty"`
	Warnings []string         `json:"warnings,omitempty"`
}

// Validate checks the protocol invariants that are independent of a tool's
// own argument schema. It is intended for API boundaries and tests.
func (call ToolCall) Validate() error {
	if !call.Name.IsValid() {
		return fmt.Errorf("unsupported tool name %q", call.Name)
	}
	if call.Arguments == nil {
		return errors.New("tool call arguments are required")
	}
	if call.ApprovalMode != "" && !call.ApprovalMode.IsValid() {
		return fmt.Errorf("unsupported approval mode %q", call.ApprovalMode)
	}
	return nil
}

// Validate checks the lifecycle and identity fields of a proposal.
func (proposal EditProposal) Validate() error {
	if proposal.ID == "" {
		return errors.New("proposal id is required")
	}
	if proposal.BaseRevision == "" {
		return errors.New("proposal base revision is required")
	}
	if len(proposal.Operations) == 0 {
		return errors.New("proposal must contain at least one operation")
	}
	if !proposal.Status.IsValid() {
		return fmt.Errorf("unsupported proposal status %q", proposal.Status)
	}
	if !isActorType(proposal.CreatedBy.ActorType) {
		return fmt.Errorf("unsupported proposal actor type %q", proposal.CreatedBy.ActorType)
	}
	for _, operation := range proposal.Operations {
		if err := operation.Validate(); err != nil {
			return fmt.Errorf("invalid proposal operation: %w", err)
		}
	}
	if proposal.ApprovalMode != "" && !proposal.ApprovalMode.IsValid() {
		return fmt.Errorf("unsupported approval mode %q", proposal.ApprovalMode)
	}
	return nil
}

// Validate checks the result envelope and its conditional error/proposal
// fields. Tool-specific result payloads are validated by the tool adapter.
func (result ToolResult) Validate() error {
	if result.ID == "" {
		return errors.New("tool result id is required")
	}
	if !result.Status.IsValid() {
		return fmt.Errorf("unsupported tool result status %q", result.Status)
	}
	if result.Name != "" && !result.Name.IsValid() {
		return fmt.Errorf("unsupported tool result name %q", result.Name)
	}
	if result.Status == ToolResultStatusError && result.Error == nil {
		return errors.New("error result must include error details")
	}
	if result.Status == ToolResultStatusPendingApproval && result.Proposal == nil {
		return errors.New("pending approval result must include a proposal")
	}
	return nil
}

func isActorType(actorType string) bool {
	switch actorType {
	case "ai", "user", "system":
		return true
	default:
		return false
	}
}
