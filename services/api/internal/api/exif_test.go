package api

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func exifPortrait(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/portrait-exif6.jpg")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestJPEGMetadataUsesEXIFDisplayDimensions(t *testing.T) {
	fixture := exifPortrait(t)
	config, _, err := image.DecodeConfig(bytes.NewReader(fixture))
	if err != nil {
		t.Fatal(err)
	}
	// This fixture has a single little-endian EXIF Orientation SHORT at byte 30.
	if len(fixture) < 38 || string(fixture[6:12]) != "Exif\x00\x00" || binary.LittleEndian.Uint16(fixture[30:32]) != 6 {
		t.Fatal("orientation fixture changed")
	}
	for _, orientation := range []uint16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9} {
		t.Run(fmt.Sprint(orientation), func(t *testing.T) {
			data := bytes.Clone(fixture)
			binary.LittleEndian.PutUint16(data[30:32], orientation)
			path := filepath.Join(t.TempDir(), "oriented.jpg")
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			meta, err := inspectImage(path)
			if err != nil {
				t.Fatal(err)
			}
			width, height := config.Width, config.Height
			if orientation >= 5 && orientation <= 8 {
				width, height = height, width
			}
			if meta.width != width || meta.height != height {
				t.Fatalf("orientation %d: got %dx%d want %dx%d", orientation, meta.width, meta.height, width, height)
			}
		})
	}
	// Invalid optional EXIF never makes an otherwise valid JPEG unimportable.
	data := bytes.Clone(fixture)
	copy(data[12:14], []byte("XX"))
	path := filepath.Join(t.TempDir(), "invalid-exif.jpg")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	meta, err := inspectImage(path)
	if err != nil || meta.width != config.Width || meta.height != config.Height {
		t.Fatalf("malformed optional EXIF did not fall back: %+v %v", meta, err)
	}
}

func TestEXIFUploadInspectAndNativeRenderAgree(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	fixture := exifPortrait(t)
	body, contentType := multipartBody(t, "portrait-exif6.jpg", fixture)
	request := httptest.NewRequest(http.MethodPost, "/api/v1/sessions/"+sess.ID+"/assets", body)
	request.Header.Set("Content-Type", contentType)
	uploaded := httptest.NewRecorder()
	h.ServeHTTP(uploaded, request)
	if uploaded.Code != 201 {
		t.Fatalf("upload: %d %s", uploaded.Code, uploaded.Body)
	}
	var a asset
	if err := json.Unmarshal(uploaded.Body.Bytes(), &a); err != nil {
		t.Fatal(err)
	}
	if a.Width != 1024 || a.Height != 820 {
		t.Fatalf("upload reports unrotated dimensions: %+v", a)
	}
	inspected := proposalRequest(t, h, "/api/v1/tools/inspect_asset/call", map[string]any{"sessionId": sess.ID})
	var result struct {
		Result struct {
			Asset asset `json:"asset"`
		} `json:"result"`
	}
	if err := json.Unmarshal(inspected.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if inspected.Code != 200 || result.Result.Asset.Width != a.Width || result.Result.Asset.Height != a.Height {
		t.Fatalf("AI inspect dimensions differ: %d %s", inspected.Code, inspected.Body)
	}
	rendered := renderedPNG(t, h, sess.ID, renderRequest{Format: "png"})
	if rendered.Bounds().Dx() != a.Width || rendered.Bounds().Dy() != a.Height {
		t.Fatalf("native dimensions %v differ from metadata %dx%d", rendered.Bounds(), a.Width, a.Height)
	}
	source := httptest.NewRecorder()
	h.ServeHTTP(source, httptest.NewRequest(http.MethodGet, "/api/v1/sessions/"+sess.ID+"/source", nil))
	if source.Code != 200 || !bytes.Equal(source.Body.Bytes(), fixture) {
		t.Fatal("reading orientation changed source JPEG bytes")
	}
}
