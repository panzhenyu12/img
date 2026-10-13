package api

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func modelMultipart(t *testing.T, imageData, mask, protection []byte, options map[string]string) (*bytes.Buffer, string) {
	t.Helper()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	for _, item := range []struct {
		name string
		data []byte
	}{{"image", imageData}, {"mask", mask}, {"protect_mask", protection}} {
		if item.data == nil {
			continue
		}
		part, err := form.CreateFormFile(item.name, item.name+".png")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = part.Write(item.data)
	}
	for name, value := range options {
		_ = form.WriteField(name, value)
	}
	_ = form.Close()
	return &body, form.FormDataContentType()
}
func grayPNG(t *testing.T, width, height int, values []byte) []byte {
	t.Helper()
	var body bytes.Buffer
	img := image.NewGray(image.Rect(0, 0, width, height))
	copy(img.Pix, values)
	if err := png.Encode(&body, img); err != nil {
		t.Fatal(err)
	}
	return body.Bytes()
}

func TestModelProxyDefaultsCacheInvalidationAndExplicitFailures(t *testing.T) {
	var calls atomic.Int32
	var failure atomic.Bool
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Error(err)
			w.WriteHeader(500)
			return
		}
		defer r.MultipartForm.RemoveAll()
		if r.FormValue("max_side") != "1024" || r.FormValue("context_mode") != "full" {
			t.Error("incorrect model defaults")
		}
		if failure.Load() {
			w.WriteHeader(503)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("X-Inpaint-Model", "lama-test")
		_, _ = w.Write(pngBytes(t))
	}))
	defer worker.Close()
	server, err := NewServer(Config{DataDir: t.TempDir(), BackgroundWorkerURL: worker.URL})
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	mask := grayPNG(t, 3, 2, []byte{255, 0, 0, 0, 0, 0})
	protect := grayPNG(t, 3, 2, make([]byte, 6))
	request := func(mask []byte, options map[string]string) *httptest.ResponseRecorder {
		body, kind := modelMultipart(t, pngBytes(t), mask, protect, options)
		r := httptest.NewRequest("POST", "/api/inpaint/model", body)
		r.Header.Set("Content-Type", kind)
		response := httptest.NewRecorder()
		server.Handler().ServeHTTP(response, r)
		return response
	}
	for index, wantCache := range []string{"miss", "hit"} {
		response := request(mask, nil)
		if response.Code != 200 || response.Header().Get("X-Inpaint-Cache") != wantCache || response.Header().Get("X-Inpaint-Model") != "lama-test" {
			t.Fatalf("request %d: %d %s", index, response.Code, response.Body)
		}
	}
	if calls.Load() != 1 {
		t.Fatal("identical candidate was inferred twice")
	}
	changed := grayPNG(t, 3, 2, []byte{0, 255, 0, 0, 0, 0})
	if response := request(changed, nil); response.Code != 200 || calls.Load() != 2 {
		t.Fatal("changed mask reused a stale candidate")
	}
	if response := request(mask, map[string]string{"mask_dilation": "8"}); response.Code != 200 || calls.Load() != 3 {
		t.Fatal("changed model config reused stale candidate")
	}
	for _, options := range []map[string]string{{"max_side": "255"}, {"context_mode": "automatic"}, {"arbitrary": "12"}} {
		if response := request(mask, options); response.Code != 422 {
			t.Fatalf("invalid model option status=%d", response.Code)
		}
	}
	if calls.Load() != 3 {
		t.Fatal("invalid inputs reached model worker")
	}
	if response := request(grayPNG(t, 2, 2, make([]byte, 4)), nil); response.Code != 422 {
		t.Fatal("dimension mismatch accepted")
	}
	failure.Store(true)
	uncached := grayPNG(t, 3, 2, []byte{0, 0, 255, 0, 0, 0})
	for range 2 {
		if response := request(uncached, nil); response.Code != 503 {
			t.Fatal("model failure silently fell back")
		}
	}
	if calls.Load() != 5 {
		t.Fatal("model failure was cached")
	}
}

func TestModelCacheEvictsOldCandidatesWithinMemoryBound(t *testing.T) {
	cache := newModelCache(6)
	var first, second, third [32]byte
	first[0] = 1
	second[0] = 2
	third[0] = 3
	cache.put(first, modelResult{data: []byte{1, 2, 3}})
	cache.put(second, modelResult{data: []byte{4, 5, 6}})
	_, _ = cache.get(first)
	cache.put(third, modelResult{data: []byte{7, 8, 9}})
	if _, ok := cache.get(second); ok {
		t.Fatal("least recently used candidate remained")
	}
	if _, ok := cache.get(first); !ok {
		t.Fatal("recent candidate was evicted")
	}
	if cache.bytes > 6 {
		t.Fatal("cache exceeded memory budget")
	}
}

func TestMixedGraphKeepsOrderFrozenPixelsAlphaAndOpacityOnce(t *testing.T) {
	var calls atomic.Int32
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_ = r.ParseMultipartForm(1 << 20)
		defer r.MultipartForm.RemoveAll()
		file, _, err := r.FormFile("image")
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		data, _ := io.ReadAll(file)
		input, err := png.Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		// Prefix horizontal flip must be visible before model inference.
		if color.NRGBAModel.Convert(input.At(0, 0)).(color.NRGBA).R != 160 {
			t.Error("model received source before prefix flip")
		}
		mask, _, _ := r.FormFile("mask")
		defer mask.Close()
		target, _ := png.Decode(mask)
		if color.GrayModel.Convert(target.At(0, 0)).(color.Gray).Y != 255 || color.GrayModel.Convert(target.At(1, 0)).(color.Gray).Y != 0 {
			t.Error("Rust did not rasterize target/protection canonically")
		}
		candidate := image.NewNRGBA(input.Bounds())
		for y := 0; y < 2; y++ {
			for x := 0; x < 3; x++ {
				candidate.SetNRGBA(x, y, color.NRGBA{200, 180, 160, 13})
			}
		}
		w.Header().Set("Content-Type", "image/png")
		_ = png.Encode(w, candidate)
	}))
	defer worker.Close()
	server, err := NewServer(Config{DataDir: t.TempDir(), BackgroundWorkerURL: worker.URL})
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPNGForTest(t, h, sess.ID)
	op := Operation{Type: "repair.inpaint", Enabled: true, Params: map[string]any{"method": "lama", "opacity": 50, "mask": map[string]any{"shape": "rect", "region": map[string]any{"x": 0, "y": 0, "width": 2.0 / 3, "height": 1}}, "protectMask": map[string]any{"shape": "rect", "region": map[string]any{"x": 1.0 / 3, "y": 0, "width": 1.0 / 3, "height": 1}}}}
	graph := EditGraph{Operations: []Operation{{Type: "flip", Enabled: true, Params: map[string]any{"axis": "horizontal"}}, op, {Type: "flip", Enabled: true, Params: map[string]any{"axis": "vertical"}}}}
	render := func(graph EditGraph) image.Image {
		result, err := server.runRender(context.Background(), sess.ID, renderRequest{EditGraph: &graph, Format: "png"}, nil)
		if err != nil {
			t.Fatal(err)
		}
		img, err := png.Decode(bytes.NewReader(result.Data))
		if err != nil {
			t.Fatal(err)
		}
		return img
	}
	img := render(graph)
	for y := 0; y < 2; y++ {
		for x := 0; x < 3; x++ {
			before := color.NRGBA{uint8((2 - x) * 80), uint8((1 - y) * 100), 30, 255}
			want := before
			if x == 0 {
				want = color.NRGBA{180, uint8((int(before.G) + 180) / 2), 95, 255}
			}
			if got := color.NRGBAModel.Convert(img.At(x, y)).(color.NRGBA); got != want {
				t.Fatalf("pixel(%d,%d)=%v want%v; order/mask/alpha/opacity contract broke", x, y, got, want)
			}
		}
	}
	// A suffix or opacity change leaves the inference input unchanged.
	graph.Operations[1].Params["opacity"] = 25
	_ = render(graph)
	if calls.Load() != 1 {
		t.Fatal("opacity change repeated model inference")
	}
	graph.Operations[1].Enabled = false
	_ = render(graph)
	if calls.Load() != 1 {
		t.Fatal("disabled LaMa contacted model")
	}
	graph.Operations[1].Params["arbitrary"] = true
	_, err = server.runRender(context.Background(), sess.ID, renderRequest{EditGraph: &graph, Format: "png"}, nil)
	if err == nil {
		t.Fatal("disabled model bypassed native validation")
	}
	delete(graph.Operations[1].Params, "arbitrary")
	graph.Operations[1].Params["execution"] = "client"
	_ = render(graph) // Disabled client nodes still validate without invoking a model.
	graph.Operations[1].Enabled = true
	graph.Operations[1].Params["opacity"] = 0
	_ = render(graph)
	graph.Operations[1].Params["opacity"] = 100
	_, err = server.runRender(context.Background(), sess.ID, renderRequest{EditGraph: &graph, Format: "png"}, nil)
	if err == nil || !strings.Contains(err.Error(), "exported in the browser") || calls.Load() != 1 {
		t.Fatalf("client mode was sent to a remote worker: calls=%d error=%v", calls.Load(), err)
	}
}
