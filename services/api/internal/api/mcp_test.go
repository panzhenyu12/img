package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image/png"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/modao/img/internal/ai"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestMCPStreamableHTTPInspectProposeAcceptUndo(t *testing.T) {
	server := newTestServer(t)
	handler := server.Handler()
	sess := createSessionForTest(t, handler)
	uploadPNGForTest(t, handler, sess.ID)
	httpServer := httptest.NewServer(handler)
	defer httpServer.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	client := mcp.NewClient(&mcp.Implementation{Name: "img-integration-test", Version: "1.0.0"}, nil)
	connection, err := client.Connect(ctx, &mcp.StreamableClientTransport{Endpoint: httpServer.URL + "/mcp", DisableStandaloneSSE: true}, nil)
	if err != nil {
		t.Fatalf("MCP initialize: %v", err)
	}
	defer connection.Close()
	listed, err := connection.ListTools(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(listed.Tools) != 8 {
		t.Fatalf("tools = %d", len(listed.Tools))
	}
	call := func(name string, args any) ai.ToolResult {
		t.Helper()
		response, err := connection.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
		if err != nil {
			t.Fatalf("MCP %s: %v", name, err)
		}
		data, err := json.Marshal(response.StructuredContent)
		if err != nil {
			t.Fatal(err)
		}
		var result ai.ToolResult
		if err := json.Unmarshal(data, &result); err != nil {
			t.Fatal(err)
		}
		if name == "render_preview" && result.Status == ai.ToolResultStatusOK {
			found := false
			for _, content := range response.Content {
				preview, ok := content.(*mcp.ImageContent)
				if !ok {
					continue
				}
				found = true
				decoded, err := png.Decode(bytes.NewReader(preview.Data))
				if err != nil || preview.MIMEType != "image/png" || decoded.Bounds().Dx() != 3 || decoded.Bounds().Dy() != 2 {
					t.Fatalf("invalid MCP preview image: mime=%s decode=%v", preview.MIMEType, err)
				}
			}
			if !found {
				t.Fatal("MCP preview returned no ImageContent")
			}
		}
		return result
	}
	inspected := call("inspect_asset", map[string]any{"sessionId": sess.ID})
	if inspected.Status != ai.ToolResultStatusOK || inspected.Result["operations"] == nil {
		t.Fatalf("inspect: %+v", inspected)
	}
	revision := inspected.Result["revision"].(string)
	proposed := call("propose_operations", map[string]any{"sessionId": sess.ID, "baseRevision": revision, "operations": []Operation{{ID: "ai-op-1", Type: "identity", Version: "1.0", Enabled: true, Params: map[string]any{}}}, "summary": "Identity review"})
	if proposed.Status != ai.ToolResultStatusPendingApproval || proposed.Proposal == nil {
		t.Fatalf("proposal: %+v", proposed)
	}
	denied := call("apply_operations", map[string]any{"sessionId": sess.ID, "proposalId": proposed.Proposal.ID, "baseRevision": revision})
	if denied.Status != ai.ToolResultStatusError {
		t.Fatal("unapproved apply succeeded")
	}
	accepted := call("apply_operations", map[string]any{"sessionId": sess.ID, "proposalId": proposed.Proposal.ID, "baseRevision": revision, "accepted": true})
	if accepted.Status != ai.ToolResultStatusOK {
		t.Fatalf("apply: %+v", accepted)
	}
	changed := accepted.Result["revision"].(string)
	if changed == revision {
		t.Fatal("apply did not advance revision")
	}
	stale := call("propose_operations", map[string]any{"sessionId": sess.ID, "baseRevision": revision, "operations": []Operation{{ID: "stale", Type: "identity", Version: "1.0", Enabled: true, Params: map[string]any{}}}})
	if stale.Status != ai.ToolResultStatusError {
		t.Fatal("stale revision was accepted")
	}
	undone := call("undo_operations", map[string]any{"sessionId": sess.ID, "baseRevision": changed, "accepted": true})
	if undone.Status != ai.ToolResultStatusOK {
		t.Fatalf("undo: %+v", undone)
	}
	current, _ := server.getSession(sess.ID)
	if len(current.EditGraph.Operations) != 0 {
		t.Fatal("undo did not restore previous graph")
	}
	preview := call("render_preview", map[string]any{"sessionId": sess.ID, "format": "png"})
	if preview.Status != ai.ToolResultStatusOK {
		t.Fatalf("preview: %+v", preview)
	}
	detection := call("detect_landmarks", map[string]any{"sessionId": sess.ID})
	if detection.Status != ai.ToolResultStatusError || detection.Error.Code != "browser_provider_required" {
		t.Fatalf("fabricated detection: %+v", detection)
	}
}

func TestGraphAssetIdentityCannotBeForged(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	response := postRender(t, h, sess.ID, `{"editGraph":{"version":"1.0","sourceAssetId":"another-session-asset","operations":[]},"format":"png"}`)
	if response.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status=%d: %s", response.Code, response.Body)
	}
}

func TestAsyncExportJobResult(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	job, err := server.startJob(sess.ID, renderRequest{Format: "png"})
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		server.mu.RLock()
		state := *server.jobs[job.ID]
		server.mu.RUnlock()
		if state.Status == "succeeded" {
			response := httptest.NewRecorder()
			h.ServeHTTP(response, httptest.NewRequest("GET", state.ResultURL, nil))
			if response.Code != 200 || response.Header().Get("Content-Type") != "image/png" {
				t.Fatalf("result status %d", response.Code)
			}
			return
		}
		if state.Status == "failed" {
			t.Fatalf("job failed: %+v", state.Error)
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("export did not complete")
}

func TestUploadDeleteRaceDoesNotPanic(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	for range 20 {
		sess := createSessionForTest(t, h)
		done := make(chan struct{})
		go func() {
			defer close(done)
			body, contentType := multipartBody(t, "race.png", pngBytes(t))
			req := httptest.NewRequest("POST", fmt.Sprintf("/api/v1/sessions/%s/assets", sess.ID), body)
			req.Header.Set("Content-Type", contentType)
			h.ServeHTTP(httptest.NewRecorder(), req)
		}()
		server.deleteSession(sess.ID)
		<-done
	}
}
