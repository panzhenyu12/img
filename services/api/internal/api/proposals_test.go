package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/modao/img/internal/ai"
)

func proposalRequest(t *testing.T, handler http.Handler, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(data))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func TestHTTPProposalRejectLifecycle(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	current, _ := server.getSession(sess.ID)
	path := "/api/v1/sessions/" + sess.ID + "/proposals"
	create := func(id string) ai.EditProposal {
		t.Helper()
		response := proposalRequest(t, h, path, map[string]any{"baseRevision": current.Revision, "operations": []Operation{{ID: id, Type: "color.basic", Enabled: true, Params: map[string]any{"exposure": 20}}}})
		if response.Code != 201 {
			t.Fatalf("propose: %d %s", response.Code, response.Body)
		}
		var proposal ai.EditProposal
		if err := json.Unmarshal(response.Body.Bytes(), &proposal); err != nil {
			t.Fatal(err)
		}
		return proposal
	}
	proposal := create("reject-me")
	for _, body := range []map[string]any{{"baseRevision": current.Revision}, {"baseRevision": current.Revision, "accepted": false}} {
		response := proposalRequest(t, h, path+"/"+proposal.ID+"/apply", body)
		if response.Code != 409 {
			t.Fatalf("unapproved apply: %d %s", response.Code, response.Body)
		}
	}
	stale := proposalRequest(t, h, path+"/"+proposal.ID+"/reject", map[string]any{"baseRevision": "stale"})
	if stale.Code != 409 {
		t.Fatal("stale rejection succeeded")
	}
	rejected := proposalRequest(t, h, path+"/"+proposal.ID+"/reject", map[string]any{"baseRevision": current.Revision})
	if rejected.Code != 200 {
		t.Fatalf("reject: %d %s", rejected.Code, rejected.Body)
	}
	var result struct {
		Proposal ai.EditProposal `json:"proposal"`
		Revision string          `json:"revision"`
		Graph    EditGraph       `json:"editGraph"`
	}
	if err := json.Unmarshal(rejected.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Proposal.Status != ai.ProposalStatusRejected || result.Revision != current.Revision || len(result.Graph.Operations) != 0 {
		t.Fatalf("reject changed edit state: %+v", result)
	}
	applied := proposalRequest(t, h, path+"/"+proposal.ID+"/apply", map[string]any{"baseRevision": current.Revision, "accepted": true})
	if applied.Code != 409 {
		t.Fatal("rejected proposal could be applied")
	}
	listed := httptest.NewRecorder()
	h.ServeHTTP(listed, httptest.NewRequest(http.MethodGet, path, nil))
	var list struct {
		Proposals []ai.EditProposal `json:"proposals"`
	}
	if err := json.Unmarshal(listed.Body.Bytes(), &list); err != nil || len(list.Proposals) != 1 || list.Proposals[0].Status != ai.ProposalStatusRejected {
		t.Fatalf("rejection was not stored: %s", listed.Body)
	}
	second := create("apply-me")
	applied = proposalRequest(t, h, path+"/"+second.ID+"/apply", map[string]any{"baseRevision": current.Revision, "accepted": true})
	if applied.Code != 200 {
		t.Fatalf("valid apply: %d %s", applied.Code, applied.Body)
	}
	current, _ = server.getSession(sess.ID)
	if response := proposalRequest(t, h, path+"/"+second.ID+"/reject", map[string]any{"baseRevision": current.Revision}); response.Code != 409 {
		t.Fatal("applied proposal could be rejected")
	}
}

func TestProposalValidatesNativeParametersBeforeStorage(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	current, _ := server.getSession(sess.ID)
	for _, op := range []Operation{
		{Type: "color.basic", Enabled: true, Params: map[string]any{"imaginary": 20}},
		{Type: "color.basic", Enabled: false, Params: map[string]any{"imaginary": 20}},
		{Type: "color.basic", Enabled: true, Params: map[string]any{"exposure": "bright"}},
		{Type: "crop", Enabled: true, Params: map[string]any{"width": 0, "height": 1}},
	} {
		response := proposalRequest(t, h, "/api/v1/sessions/"+sess.ID+"/proposals", map[string]any{"baseRevision": current.Revision, "operations": []Operation{op}})
		if response.Code != 409 || !strings.Contains(response.Body.String(), "proposal_rejected") {
			t.Fatalf("invalid native parameters became a proposal: %d %s", response.Code, response.Body)
		}
		if strings.Contains(response.Body.String(), server.dataDir) {
			t.Fatal("proposal validation leaked a server path")
		}
	}
	server.mu.RLock()
	defer server.mu.RUnlock()
	state := server.sessions[sess.ID]
	if len(state.proposals) != 0 || state.Revision != current.Revision || len(state.EditGraph.Operations) != 0 {
		t.Fatal("invalid proposal changed session state")
	}
}

func TestRejectDuringApplyCannotCommit(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	current, _ := server.getSession(sess.ID)
	proposal, err := server.propose(context.Background(), sess.ID, current.Revision, []Operation{{Type: "color.basic", Enabled: true, Params: map[string]any{"exposure": 20}}}, "")
	if err != nil {
		t.Fatal(err)
	}
	gateDir := t.TempDir()
	started := filepath.Join(gateDir, "started")
	release := filepath.Join(gateDir, "release")
	wrapper := filepath.Join(gateDir, "worker")
	quote := func(value string) string { return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'" }
	script := "#!/bin/sh\n: > " + quote(started) + "\nwhile [ ! -f " + quote(release) + " ]; do sleep 0.01; done\nexec " + quote(server.renderBinary) + " \"$@\"\n"
	if err := os.WriteFile(wrapper, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	server.renderBinary = wrapper
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	t.Cleanup(func() { _ = os.WriteFile(release, nil, 0o600) })
	done := make(chan error, 1)
	go func() {
		_, _, err := server.applyProposal(ctx, sess.ID, proposal.ID, current.Revision, true)
		done <- err
	}()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if _, err := os.Stat(started); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("apply did not enter Rust validation")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, _, _, err := server.rejectProposal(sess.ID, proposal.ID, current.Revision); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(release, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err == nil {
		t.Fatal("application committed a proposal rejected during rendering")
	}
	state, _ := server.getSession(sess.ID)
	if state.Revision != current.Revision || len(state.EditGraph.Operations) != 0 {
		t.Fatal("rejection race changed the edit graph")
	}
}
