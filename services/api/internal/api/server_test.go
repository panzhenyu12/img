package api

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSessionUploadPreviewAndRenderIdentity(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()

	create := httptest.NewRecorder()
	h.ServeHTTP(create, httptest.NewRequest(http.MethodPost, "/api/v1/sessions", nil))
	if create.Code != http.StatusCreated {
		t.Fatalf("create status = %d, body = %s", create.Code, create.Body.String())
	}
	var sess session
	if err := json.Unmarshal(create.Body.Bytes(), &sess); err != nil {
		t.Fatalf("decode session: %v", err)
	}
	if sess.ID == "" {
		t.Fatal("session id is empty")
	}

	original := pngBytes(t)
	body, contentType := multipartBody(t, "portrait.png", original)
	upload := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/"+sess.ID+"/assets", body)
	req.Header.Set("Content-Type", contentType)
	h.ServeHTTP(upload, req)
	if upload.Code != http.StatusCreated {
		t.Fatalf("upload status = %d, body = %s", upload.Code, upload.Body.String())
	}
	var a asset
	if err := json.Unmarshal(upload.Body.Bytes(), &a); err != nil {
		t.Fatalf("decode asset: %v", err)
	}
	if a.Width != 3 || a.Height != 2 || a.MimeType != "image/png" || a.Size != int64(len(original)) {
		t.Fatalf("unexpected asset metadata: %+v", a)
	}

	for _, path := range []string{"/source", "/image", "/preview", "/render"} {
		var got *httptest.ResponseRecorder
		if path == "/preview" || path == "/render" {
			got = httptest.NewRecorder()
			r := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/"+sess.ID+path, strings.NewReader(`{"editGraph":{"version":"1.0","operations":[{"type":"identity"}]}}`))
			r.Header.Set("Content-Type", "application/json")
			h.ServeHTTP(got, r)
		} else {
			got = httptest.NewRecorder()
			h.ServeHTTP(got, httptest.NewRequest(http.MethodGet, "/api/v1/sessions/"+sess.ID+path, nil))
		}
		if got.Code != http.StatusOK {
			t.Fatalf("%s status = %d, body = %s", path, got.Code, got.Body.String())
		}
		if got.Header().Get("Content-Type") != "image/png" {
			t.Fatalf("%s content type = %q", path, got.Header().Get("Content-Type"))
		}
		if path == "/source" || path == "/image" {
			if !bytes.Equal(got.Body.Bytes(), original) {
				t.Fatalf("%s changed source bytes", path)
			}
		} else {
			decoded, err := png.Decode(bytes.NewReader(got.Body.Bytes()))
			if err != nil {
				t.Fatal(err)
			}
			baseline, err := png.Decode(bytes.NewReader(original))
			if err != nil || !imagesEqual(decoded, baseline) {
				t.Fatalf("%s changed identity pixels: %v", path, err)
			}
		}
	}
}

func TestRenderRejectsUnsupportedOperation(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/"+sess.ID+"/render", strings.NewReader(`{"operation":{"type":"warp.face"}}`))
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()
	h.ServeHTTP(res, req)
	if res.Code != http.StatusUnprocessableEntity {
		t.Fatalf("status = %d, body = %s", res.Code, res.Body.String())
	}
}

func TestRenderBasicAdjustChangesPixels(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	body := `{"editGraph":{"version":"1.0","operations":[{"id":"bright","type":"color.basic","enabled":true,"params":{"exposure":40}}]},"format":"png"}`
	res := postRender(t, h, sess.ID, body)
	if res.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", res.Code, res.Body.String())
	}
	img, err := png.Decode(bytes.NewReader(res.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode rendered png: %v", err)
	}
	if got := color.RGBAModel.Convert(img.At(2, 1)).(color.RGBA); got.R <= 180 || got.G <= 110 {
		t.Fatalf("expected exposure to brighten pixel, got %#v", got)
	}
}

func TestRenderFilmFilterAndCrop(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	body := `{"editGraph":{"version":"1.0","operations":[{"id":"crop","type":"crop","enabled":true,"params":{"x":1,"y":0,"width":2,"height":1}},{"id":"film","type":"filter.film","enabled":true,"params":{"preset":"classic","intensity":1,"grain":0.1,"vignette":0.2}}]},"format":"png"}`
	res := postRender(t, h, sess.ID, body)
	if res.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", res.Code, res.Body.String())
	}
	img, err := png.Decode(bytes.NewReader(res.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode rendered png: %v", err)
	}
	if got := img.Bounds().Size(); got.X != 2 || got.Y != 1 {
		t.Fatalf("crop dimensions = %v, want 2x1", got)
	}
}

func TestRenderRotateAndFlip(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	rotated := postRender(t, h, sess.ID, `{"editGraph":{"operations":[{"type":"rotate","params":{"degrees":90}}]},"format":"png"}`)
	if rotated.Code != http.StatusOK {
		t.Fatalf("rotate status = %d, body = %s", rotated.Code, rotated.Body.String())
	}
	img, err := png.Decode(bytes.NewReader(rotated.Body.Bytes()))
	if err != nil {
		t.Fatalf("decode rotated png: %v", err)
	}
	if got := img.Bounds().Size(); got.X != 2 || got.Y != 3 {
		t.Fatalf("rotate dimensions = %v, want 2x3", got)
	}
	flipped := postRender(t, h, sess.ID, `{"editGraph":{"operations":[{"type":"flip","params":{"axis":"horizontal"}}]},"format":"png"}`)
	if flipped.Code != http.StatusOK {
		t.Fatalf("flip status = %d, body = %s", flipped.Code, flipped.Body.String())
	}
}

func TestRenderDisabledOperationIsNoop(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	original := uploadPNGForTest(t, h, sess.ID)
	body := `{"editGraph":{"version":"1.0","operations":[{"id":"disabled","type":"color.basic","enabled":false,"params":{"exposure":100}}]},"format":"png"}`
	res := postRender(t, h, sess.ID, body)
	if res.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", res.Code, res.Body.String())
	}
	if !bytes.Equal(res.Body.Bytes(), original) {
		t.Fatal("disabled operation changed the source bytes")
	}
}

func postRender(t *testing.T, h http.Handler, sessionID, body string) *httptest.ResponseRecorder {
	t.Helper()
	res := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/"+sessionID+"/render", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	h.ServeHTTP(res, req)
	return res
}

func TestUploadRejectsNonImageAndOversize(t *testing.T) {
	dataDir := t.TempDir()
	server, err := NewServer(Config{DataDir: dataDir, MaxUploadBytes: 8})
	if err != nil {
		t.Fatal(err)
	}
	h := server.Handler()
	sess := createSessionForTest(t, h)
	tooLarge := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/"+sess.ID+"/upload", bytes.NewReader(bytes.Repeat([]byte("x"), 9)))
	tooLarge.Header.Set("Content-Type", "application/octet-stream")
	res := httptest.NewRecorder()
	h.ServeHTTP(res, tooLarge)
	if res.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversize status = %d, body = %s", res.Code, res.Body.String())
	}
}

func TestSessionExpires(t *testing.T) {
	dataDir := t.TempDir()
	now := time.Now()
	server, err := NewServer(Config{DataDir: dataDir, SessionTTL: time.Second, Now: func() time.Time { return now }})
	if err != nil {
		t.Fatal(err)
	}
	h := server.Handler()
	sess := createSessionForTest(t, h)
	now = now.Add(2 * time.Second)
	res := httptest.NewRecorder()
	h.ServeHTTP(res, httptest.NewRequest(http.MethodGet, "/api/v1/sessions/"+sess.ID, nil))
	if res.Code != http.StatusNotFound {
		t.Fatalf("expired session status = %d", res.Code)
	}
}

func newTestServer(t *testing.T) *Server {
	t.Helper()
	server, err := NewServer(Config{DataDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(server.Close)
	return server
}

func createSessionForTest(t *testing.T, h http.Handler) session {
	t.Helper()
	res := httptest.NewRecorder()
	h.ServeHTTP(res, httptest.NewRequest(http.MethodPost, "/api/v1/sessions", nil))
	if res.Code != http.StatusCreated {
		t.Fatalf("create status = %d", res.Code)
	}
	var sess session
	if err := json.Unmarshal(res.Body.Bytes(), &sess); err != nil {
		t.Fatal(err)
	}
	return sess
}

func uploadPNGForTest(t *testing.T, h http.Handler, sessionID string) []byte {
	t.Helper()
	body, contentType := multipartBody(t, "image.png", pngBytes(t))
	res := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/"+sessionID+"/upload", body)
	req.Header.Set("Content-Type", contentType)
	h.ServeHTTP(res, req)
	if res.Code != http.StatusCreated {
		t.Fatalf("upload status = %d, body = %s", res.Code, res.Body.String())
	}
	return pngBytes(t)
}

func pngBytes(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 3, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 3; x++ {
			img.Set(x, y, color.RGBA{uint8(x * 80), uint8(y * 100), 30, 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func multipartBody(t *testing.T, filename string, data []byte) (*bytes.Buffer, string) {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", filepath.Base(filename))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return &body, writer.FormDataContentType()
}
