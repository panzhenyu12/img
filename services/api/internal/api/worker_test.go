package api

import (
	"bytes"
	"context"
	"errors"
	"image/color"
	"image/png"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func slowWorker(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "slow-worker")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nexec sleep 60\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestNativeWorkerDeadlineAndQueuedJobCancellation(t *testing.T) {
	server, err := NewServer(Config{DataDir: t.TempDir(), RenderBinary: slowWorker(t), RenderTimeout: 100 * time.Millisecond, RenderConcurrency: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	handler := server.Handler()
	sess := createSessionForTest(t, handler)
	uploadPNGForTest(t, handler, sess.ID)
	request := renderRequest{Operation: &Operation{ID: "x", Version: "1.0", Type: "rotate", Enabled: true, Params: map[string]any{"degrees": 15}}, Format: "png"}
	_, err = server.runRender(context.Background(), sess.ID, request, nil)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline, got %v", err)
	}
	server.renderTimeout = 10 * time.Second
	jobs := []renderJob{}
	for range 3 {
		job, err := server.startJob(sess.ID, request)
		if err != nil {
			t.Fatal(err)
		}
		jobs = append(jobs, job)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		server.mu.RLock()
		running, queued := 0, 0
		for _, job := range server.jobs {
			if job.Status == "running" {
				running++
			}
			if job.Status == "queued" {
				queued++
			}
		}
		server.mu.RUnlock()
		if running == 1 && queued == 2 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	server.mu.RLock()
	running := 0
	for _, job := range server.jobs {
		if job.Status == "running" {
			running++
		}
	}
	server.mu.RUnlock()
	if running != 1 {
		t.Fatalf("bounded concurrency: running=%d", running)
	}
	for _, job := range jobs {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest("POST", "/api/v1/jobs/"+job.ID+"/cancel", nil))
		if response.Code != 200 {
			t.Fatalf("cancel status %d", response.Code)
		}
	}
	server.wg.Wait()
	server.mu.RLock()
	defer server.mu.RUnlock()
	for _, job := range server.jobs {
		if job.Status != "cancelled" {
			t.Fatalf("cancelled job status=%s", job.Status)
		}
	}
	entries, err := os.ReadDir(server.dataDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			t.Fatalf("worker temporary directory leaked: %s", entry.Name())
		}
	}
}

func TestNativeWorkerRejectsInvalidParametersWithoutPathLeak(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	for _, body := range []string{
		`{"operation":{"type":"color.basic","params":{"inventedAdjustment":60}},"format":"png"}`,
		`{"operation":{"type":"crop","params":{"width":0,"height":1}},"format":"png"}`,
	} {
		result := postRender(t, h, sess.ID, body)
		if result.Code != 422 {
			t.Fatalf("invalid params status=%d, body=%s", result.Code, result.Body)
		}
		if strings.Contains(result.Body.String(), server.dataDir) {
			t.Fatal("render error leaked temporary server path")
		}
	}
}

func TestSameFormatNoopNodesRejectInvalidParameters(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	original := uploadPNGForTest(t, h, sess.ID)
	for _, item := range []struct {
		name string
		body string
	}{
		{"disabled unknown key", `{"operation":{"type":"color.basic","enabled":false,"params":{"imaginary":20}},"format":"png"}`},
		{"disabled invalid range", `{"operation":{"type":"color.basic","enabled":false,"params":{"exposure":500}},"format":"png"}`},
		{"identity unknown key", `{"operation":{"type":"identity","enabled":true,"params":{"imaginary":20}},"format":"png"}`},
	} {
		t.Run(item.name, func(t *testing.T) {
			response := postRender(t, h, sess.ID, item.body)
			if response.Code != 422 {
				t.Fatalf("same-format no-op bypassed validation: %d %s", response.Code, response.Body)
			}
			if strings.Contains(response.Body.String(), server.dataDir) {
				t.Fatal("validation leaked a server path")
			}
		})
	}
	source := httptest.NewRecorder()
	h.ServeHTTP(source, httptest.NewRequest("GET", "/api/v1/sessions/"+sess.ID+"/source", nil))
	if source.Code != 200 || !bytes.Equal(source.Body.Bytes(), original) {
		t.Fatal("invalid no-op request changed the source")
	}
}

func TestEnabledIdentityKeepsNativeBlendEffect(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	original := uploadPNGForTest(t, h, sess.ID)
	baseline, err := png.Decode(bytes.NewReader(original))
	if err != nil {
		t.Fatal(err)
	}
	response := postRender(t, h, sess.ID, `{"operation":{"type":"identity","enabled":true,"params":{"blendMode":"multiply"}},"format":"png"}`)
	if response.Code != 200 {
		t.Fatalf("identity multiply: %d %s", response.Code, response.Body)
	}
	img, err := png.Decode(bytes.NewReader(response.Body.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	got := color.RGBAModel.Convert(img.At(2, 1)).(color.RGBA)
	before := color.RGBAModel.Convert(baseline.At(2, 1)).(color.RGBA)
	if got.R >= before.R || got.G >= before.G || got.B >= before.B {
		t.Fatalf("native identity multiply effect was discarded: before=%#v after=%#v", before, got)
	}
}

func TestOnlyEmptyGraphCanBypassUnavailableWorker(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	original := uploadPNGForTest(t, h, sess.ID)
	server.renderBinary = filepath.Join(t.TempDir(), "missing-render-worker")
	empty := postRender(t, h, sess.ID, `{"editGraph":{"operations":[]},"format":"png"}`)
	if empty.Code != 200 || !bytes.Equal(empty.Body.Bytes(), original) {
		t.Fatalf("empty graph copy: %d %s", empty.Code, empty.Body)
	}
	for _, body := range []string{
		`{"operation":{"type":"identity","params":{}},"format":"png"}`,
		`{"operation":{"type":"color.basic","enabled":false,"params":{"exposure":20}},"format":"png"}`,
	} {
		response := postRender(t, h, sess.ID, body)
		if response.Code != 503 {
			t.Fatalf("node graph skipped required worker: %d %s", response.Code, response.Body)
		}
	}
}
