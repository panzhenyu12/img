package ai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestToolCallRoundTripAndValidation(t *testing.T) {
	want := ToolCall{
		ID:           "call-1",
		Name:         ToolProposeOperations,
		Arguments:    map[string]any{"prompt": "slim the face", "strength": 0.25},
		ApprovalMode: ApprovalModeConfirm,
		SessionID:    "session-1",
		RequestID:    "request-1",
	}
	payload, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal tool call: %v", err)
	}
	var got ToolCall
	if err := json.Unmarshal(payload, &got); err != nil {
		t.Fatalf("unmarshal tool call: %v", err)
	}
	if err := got.Validate(); err != nil {
		t.Fatalf("validate tool call: %v", err)
	}
	if got.Name != want.Name || got.ApprovalMode != want.ApprovalMode || got.Arguments["prompt"] != "slim the face" {
		t.Fatalf("round trip mismatch: %#v", got)
	}
}

func TestEditProposalAndPendingResultRoundTrip(t *testing.T) {
	proposal := EditProposal{
		ID:           "proposal-1",
		BaseRevision: "rev-7",
		Operations: []Operation{{
			ID:      "op-1",
			Type:    "face.adjust",
			Version: "1.0",
			Enabled: true,
			Params:  map[string]any{"faceSlim": 18},
		}},
		Status:       ProposalStatusProposed,
		ApprovalMode: ApprovalModeConfirm,
		CreatedBy: ProposalActor{
			ActorType:  "ai",
			ActorID:    "deepagent",
			Model:      "demo-model",
			ToolCallID: "call-1",
		},
		Summary:  "Apply a small face slimming adjustment",
		Warnings: []string{"Review the jaw edge before applying"},
	}
	if err := proposal.Validate(); err != nil {
		t.Fatalf("validate proposal: %v", err)
	}

	want := ToolResult{
		ID:       "call-1",
		Status:   ToolResultStatusPendingApproval,
		Name:     ToolProposeOperations,
		Proposal: &proposal,
	}
	payload, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("marshal tool result: %v", err)
	}
	var got ToolResult
	if err := json.Unmarshal(payload, &got); err != nil {
		t.Fatalf("unmarshal tool result: %v", err)
	}
	if err := got.Validate(); err != nil {
		t.Fatalf("validate tool result: %v", err)
	}
	if got.Proposal == nil || got.Proposal.Operations[0].Type != "face.adjust" {
		t.Fatalf("proposal was not preserved: %#v", got)
	}
	if !strings.Contains(string(payload), `"approvalMode":"confirm"`) {
		t.Fatalf("approval mode missing from payload: %s", payload)
	}
}

func TestProtocolValidationRejectsUnsafeOrIncompleteMessages(t *testing.T) {
	if err := (ToolCall{Name: ToolName("run_shell"), Arguments: map[string]any{}}).Validate(); err == nil {
		t.Fatal("unsupported tool name should be rejected")
	}
	if err := (ToolCall{Name: ToolRenderPreview}).Validate(); err == nil {
		t.Fatal("missing arguments should be rejected")
	}
	if err := (ToolResult{ID: "call-1", Status: ToolResultStatusError}).Validate(); err == nil {
		t.Fatal("error result without details should be rejected")
	}
	if err := (ToolResult{ID: "call-1", Status: ToolResultStatusPendingApproval}).Validate(); err == nil {
		t.Fatal("pending result without proposal should be rejected")
	}
	if err := (EditProposal{
		ID:           "proposal-1",
		BaseRevision: "rev-1",
		Status:       ProposalStatusProposed,
		CreatedBy:    ProposalActor{ActorType: "ai"},
	}).Validate(); err == nil {
		t.Fatal("proposal without operations should be rejected")
	}
	if err := (Operation{ID: "op-1", Type: "run_shell", Version: "1.0", Enabled: true, Params: map[string]any{}}).Validate(); err == nil {
		t.Fatal("unsupported operation should be rejected")
	}
}
