package api

import (
	"bytes"
	"encoding/json"
	"github.com/modao/img/internal/ai"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func uploadPortrait(t *testing.T, h http.Handler, sessionID string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("../../../../apps/web/tests/fixtures/portrait.jpg"))
	if err != nil {
		t.Fatalf("read shared portrait fixture: %v", err)
	}
	body, contentType := multipartBody(t, "portrait.jpg", data)
	req := httptest.NewRequest("POST", "/api/v1/sessions/"+sessionID+"/assets", body)
	req.Header.Set("Content-Type", contentType)
	res := httptest.NewRecorder()
	h.ServeHTTP(res, req)
	if res.Code != 201 {
		t.Fatalf("upload portrait: %d %s", res.Code, res.Body)
	}
	return data
}

func renderedPNG(t *testing.T, h http.Handler, sessionID string, req renderRequest) image.Image {
	t.Helper()
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	result := postRender(t, h, sessionID, string(data))
	if result.Code != 200 {
		t.Fatalf("render status=%d: %s", result.Code, result.Body)
	}
	img, err := png.Decode(bytes.NewReader(result.Body.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func imagesEqual(a, b image.Image) bool {
	if a.Bounds().Size() != b.Bounds().Size() {
		return false
	}
	for y := 0; y < a.Bounds().Dy(); y++ {
		for x := 0; x < a.Bounds().Dx(); x++ {
			if color.NRGBAModel.Convert(a.At(x, y)) != color.NRGBAModel.Convert(b.At(x, y)) {
				return false
			}
		}
	}
	return true
}

// This captured localization accompanies portrait.jpg. The face polygon and
// feature contours are normalized; the subject matte is losslessly stored as
// grayscale PNG. It is protocol input, not a ground-truth naturalness score.
func portraitLocalization(t *testing.T) (map[string]any, map[string]any) {
	t.Helper()
	data, err := os.ReadFile("testdata/portrait-localization.json")
	if err != nil {
		t.Fatal(err)
	}
	var params map[string]any
	if err := json.Unmarshal(data, &params); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open("testdata/portrait-subject-mask.png")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	matte, err := png.Decode(file)
	if err != nil {
		t.Fatal(err)
	}
	alpha := make([]int, 0, matte.Bounds().Dx()*matte.Bounds().Dy())
	for y := 0; y < matte.Bounds().Dy(); y++ {
		for x := 0; x < matte.Bounds().Dx(); x++ {
			alpha = append(alpha, int(color.GrayModel.Convert(matte.At(x, y)).(color.Gray).Y))
		}
	}
	return params, map[string]any{"shape": "raster", "width": matte.Bounds().Dx(), "height": matte.Bounds().Dy(), "data": alpha}
}

// Every operation is exercised through HTTP against the same real photograph
// and actual Rust CLI. This catches protocol drift across independent clients,
// unsupported declarations, no-op placeholders, and accidental source writes.
// This is an HTTP/CLI smoke test. Naturalness and anatomical effect are
// separately assessed with contour/texture tests and visual comparisons.
func TestEveryNativeOperationOnRealPortrait(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	sourceBytes := uploadPortrait(t, h, sess.ID)
	current, _ := server.getSession(sess.ID)
	baseline := renderedPNG(t, h, sess.ID, renderRequest{EditGraph: &EditGraph{Version: "1.0", SourceAssetID: current.Image.ID}, Format: "png"})
	region := map[string]any{"x": 0.2, "y": 0.12, "width": 0.6, "height": 0.68}
	face, subject := portraitLocalization(t)
	face["adjustments"] = map[string]any{"faceWidth": 60}
	feature := map[string]any{"region": face["region"], "semanticMasks": face["semanticMasks"], "landmarks": face["landmarks"], "adjustments": map[string]any{"eyeSize": 60}}
	arm := map[string]any{
		"bodyPart": "left-upper-arm", "region": map[string]any{"x": 0.61, "y": 0.36, "width": 0.25, "height": 0.31},
		"landmarks":   []map[string]any{{"x": 0.75, "y": 0.42}, {"x": 0.75, "y": 0.65}},
		"subjectMask": subject, "adjustments": map[string]any{"arms": 50},
	}
	stroke := map[string]any{"x": 0.5, "y": 0.5, "sourceX": 0.08, "sourceY": 0.15, "radius": 0.04, "strength": 100}
	lut := [][]float64{}
	for b := 0; b < 2; b++ {
		for g := 0; g < 2; g++ {
			for r := 0; r < 2; r++ {
				lut = append(lut, []float64{1 - float64(r), 1 - float64(g), 1 - float64(b)})
			}
		}
	}
	cases := []struct {
		kind   string
		params map[string]any
		points []ai.ControlPoint
		size   image.Point
	}{
		{"identity", map[string]any{}, nil, baseline.Bounds().Size()},
		{"crop", map[string]any{"x": 20, "y": 20, "width": 200, "height": 300}, nil, image.Pt(200, 300)},
		{"resize", map[string]any{"width": 200, "height": 250}, nil, image.Pt(200, 250)},
		{"rotate", map[string]any{"degrees": 90}, nil, image.Pt(baseline.Bounds().Dy(), baseline.Bounds().Dx())},
		{"flip", map[string]any{"axis": "horizontal"}, nil, baseline.Bounds().Size()},
		{"perspective", map[string]any{"corners": []map[string]any{{"x": 0.03, "y": 0.02}, {"x": 0.97, "y": 0.04}, {"x": 1, "y": 0.98}, {"x": 0, "y": 1}}}, nil, baseline.Bounds().Size()},
		{"warp.grid", map[string]any{}, []ai.ControlPoint{{X: 0, Y: 0}, {X: 1, Y: 0}, {X: 0, Y: 1}, {X: 1, Y: 1}, {X: 0.5, Y: 0.5, InX: 0.5, InY: 0.5, OutX: 0.57, OutY: 0.5}}, baseline.Bounds().Size()},
		{"warp.liquify", map[string]any{"strokes": []map[string]any{{"x": 0.5, "y": 0.5, "toX": 0.55, "toY": 0.5, "radius": 0.13, "strength": 80, "mode": "push"}}}, nil, baseline.Bounds().Size()},
		{"face.adjust", face, nil, baseline.Bounds().Size()},
		{"feature.adjust", feature, nil, baseline.Bounds().Size()},
		{"skin.retouch", map[string]any{"region": region, "smooth": 50, "whiten": 40}, nil, baseline.Bounds().Size()},
		{"body.adjust", arm, nil, baseline.Bounds().Size()},
		{"clothing.adjust", map[string]any{"region": map[string]any{"x": 0.55, "y": 0.4, "width": 0.21, "height": 0.2}, "adjustments": map[string]any{"brightness": 40}}, nil, baseline.Bounds().Size()},
		{"repair.clone", map[string]any{"strokes": []map[string]any{stroke}}, nil, baseline.Bounds().Size()},
		{"repair.heal", map[string]any{"strokes": []map[string]any{stroke}}, nil, baseline.Bounds().Size()},
		{"repair.inpaint", map[string]any{"region": map[string]any{"x": 0.4, "y": 0.4, "width": 0.03, "height": 0.03}, "radius": 0.03, "iterations": 3}, nil, baseline.Bounds().Size()},
		{"paint.brush", map[string]any{"color": []int{255, 0, 0}, "strokes": []map[string]any{{"x": 0.5, "y": 0.5, "radius": 0.04, "strength": 100}}}, nil, baseline.Bounds().Size()},
		{"color.basic", map[string]any{"exposure": 40}, nil, baseline.Bounds().Size()},
		{"color.curves", map[string]any{"points": []map[string]any{{"x": 0, "y": 0}, {"x": 0.5, "y": 0.7}, {"x": 1, "y": 1}}}, nil, baseline.Bounds().Size()},
		{"color.hsl", map[string]any{"hue": 40, "saturation": 20}, nil, baseline.Bounds().Size()},
		{"filter.film", map[string]any{"preset": "classic", "intensity": 80, "grain": 30, "seed": 7}, nil, baseline.Bounds().Size()},
		{"filter.lut", map[string]any{"size": 2, "data": lut, "amount": 100}, nil, baseline.Bounds().Size()},
	}
	for _, item := range cases {
		t.Run(item.kind, func(t *testing.T) {
			op := Operation{ID: item.kind, Type: item.kind, Version: "1.0", Enabled: true, Params: item.params, ControlPoints: item.points}
			result := renderedPNG(t, h, sess.ID, renderRequest{EditGraph: &EditGraph{Version: "1.0", SourceAssetID: current.Image.ID, Operations: []Operation{op}}, Format: "png"})
			if result.Bounds().Size() != item.size {
				t.Fatalf("dimensions=%v want %v", result.Bounds().Size(), item.size)
			}
			if item.kind == "identity" {
				if !imagesEqual(result, baseline) {
					t.Fatal("identity changed pixels")
				}
			} else if imagesEqual(result, baseline) {
				t.Fatal("enabled operation produced no pixel or geometry change")
			}
		})
	}
	source := httptest.NewRecorder()
	h.ServeHTTP(source, httptest.NewRequest("GET", "/api/v1/sessions/"+sess.ID+"/source", nil))
	if !bytes.Equal(source.Body.Bytes(), sourceBytes) {
		t.Fatal("rendering changed source JPEG bytes")
	}
}

func TestSemanticEditsRejectMissingLocalizationAndPendingAlgorithms(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPortrait(t, h, sess.ID)
	for _, item := range []struct {
		name, kind, reason string
		params             map[string]any
	}{
		{"face-rectangle-only", "face.adjust", "face contour", map[string]any{"region": map[string]any{"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.4}, "adjustments": map[string]any{"faceWidth": 40}}},
		{"body-without-subject", "body.adjust", "subjectMask", map[string]any{"adjustments": map[string]any{"waist": 40}}},
		{"body-without-landmarks", "body.adjust", "landmarks", map[string]any{"bodyPart": "waist", "subjectMask": map[string]any{"shape": "rect", "region": map[string]any{"x": 0.2, "y": 0.4, "width": 0.6, "height": 0.5}}, "adjustments": map[string]any{"waist": 40}}},
		{"eye-bag-pending", "feature.adjust", "no validated algorithm", map[string]any{"region": map[string]any{"x": 0.3, "y": 0.1, "width": 0.4, "height": 0.3}, "adjustments": map[string]any{"eyeBag": 40}}},
		{"clothing-fit-pending", "clothing.adjust", "no validated algorithm", map[string]any{"region": map[string]any{"x": 0.2, "y": 0.4, "width": 0.6, "height": 0.5}, "adjustments": map[string]any{"fit": 40}}},
	} {
		t.Run(item.name, func(t *testing.T) {
			body, err := json.Marshal(renderRequest{Operation: &Operation{Type: item.kind, Enabled: true, Params: item.params}, Format: "png"})
			if err != nil {
				t.Fatal(err)
			}
			res := postRender(t, h, sess.ID, string(body))
			if res.Code != http.StatusUnprocessableEntity || !strings.Contains(res.Body.String(), item.reason) {
				t.Fatalf("status=%d body=%s; want rejected semantic input mentioning %q", res.Code, res.Body, item.reason)
			}
		})
	}
}

func TestRepairAndWarpRespectProtectedMask(t *testing.T) {
	server := newTestServer(t)
	h := server.Handler()
	sess := createSessionForTest(t, h)
	uploadPortrait(t, h, sess.ID)
	baseline := renderedPNG(t, h, sess.ID, renderRequest{EditGraph: &EditGraph{Operations: []Operation{}}, Format: "png"})
	mask := map[string]any{"kind": "rect", "region": map[string]any{"x": 0, "y": 0, "width": 0.5, "height": 1}}
	for _, item := range []struct {
		kind   string
		params map[string]any
	}{
		{"color.basic", map[string]any{"exposure": 60, "protectMask": mask}},
		{"warp.liquify", map[string]any{"strokes": []map[string]any{{"x": 0.5, "y": 0.5, "toX": 0.6, "toY": 0.5, "radius": 0.3, "strength": 100}}, "protectMask": mask}},
		{"repair.clone", map[string]any{"strokes": []map[string]any{{"x": 0.5, "y": 0.5, "sourceX": 0.05, "sourceY": 0.1, "radius": 0.2}}, "protectMask": mask}},
		{"repair.heal", map[string]any{"strokes": []map[string]any{{"x": 0.5, "y": 0.5, "sourceX": 0.05, "sourceY": 0.1, "radius": 0.2}}, "protectMask": mask}},
		{"repair.inpaint", map[string]any{"region": map[string]any{"x": 0.45, "y": 0.4, "width": 0.1, "height": 0.04}, "iterations": 3, "protectMask": mask}},
		{"paint.brush", map[string]any{"color": []int{255, 0, 0}, "strokes": []map[string]any{{"x": 0.5, "y": 0.5, "radius": 0.2, "strength": 100}}, "protectMask": mask}},
	} {
		t.Run(item.kind, func(t *testing.T) {
			result := renderedPNG(t, h, sess.ID, renderRequest{EditGraph: &EditGraph{Operations: []Operation{{Type: item.kind, Enabled: true, Params: item.params}}}, Format: "png"})
			changed := false
			for y := 0; y < baseline.Bounds().Dy(); y++ {
				for x := 0; x < baseline.Bounds().Dx(); x++ {
					same := color.NRGBAModel.Convert(baseline.At(x, y)) == color.NRGBAModel.Convert(result.At(x, y))
					if x < baseline.Bounds().Dx()/2 && !same {
						t.Fatalf("protected pixel (%d,%d) changed", x, y)
					}
					if x >= baseline.Bounds().Dx()/2 && !same {
						changed = true
					}
				}
			}
			if !changed {
				t.Fatal("unprotected region did not change")
			}
		})
	}
}
