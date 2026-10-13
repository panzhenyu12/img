package api

import (
	"bytes"
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/rwcarlsen/goexif/exif"
)

var errBackgroundUnavailable = errors.New("local LaMa worker is unavailable; start the background worker or select manual exemplar inpainting")

// Only successful model candidates are cached, before Rust opacity/blending.
// A changed suffix graph or layer opacity can reuse the same inference result.
type modelResult struct {
	data    []byte
	headers http.Header
}
type modelEntry struct {
	key    [32]byte
	result modelResult
}
type modelCache struct {
	mu           sync.Mutex
	items        map[[32]byte]*list.Element
	order        *list.List
	bytes, limit int
}

func newModelCache(limit int) *modelCache {
	return &modelCache{items: make(map[[32]byte]*list.Element), order: list.New(), limit: limit}
}
func (c *modelCache) get(key [32]byte) (modelResult, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.items[key]
	if !ok {
		return modelResult{}, false
	}
	c.order.MoveToFront(entry)
	return entry.Value.(modelEntry).result, true
}
func (c *modelCache) put(key [32]byte, result modelResult) {
	if len(result.data) > c.limit {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if entry, ok := c.items[key]; ok {
		c.order.MoveToFront(entry)
		return
	}
	for c.bytes+len(result.data) > c.limit {
		entry := c.order.Back()
		value := entry.Value.(modelEntry)
		delete(c.items, value.key)
		c.bytes -= len(value.result.data)
		c.order.Remove(entry)
	}
	c.items[key] = c.order.PushFront(modelEntry{key: key, result: result})
	c.bytes += len(result.data)
}

var modelHeaderNames = []string{"X-Inpaint-Model", "X-Inpaint-Device", "X-Inpaint-Inference-Ms", "X-Inpaint-Total-Ms", "X-Inpaint-Peak-Rss-Mb", "X-Inpaint-Selected-Pixels", "X-Inpaint-Seam-Blend", "X-Inpaint-Context-Mode"}

func modelOptions(values map[string]string) (map[string]string, error) {
	result := map[string]string{"max_side": "1024", "context_mode": "full", "context_padding": "256", "mask_dilation": "4", "seam_blend": "none"}
	for key, value := range values {
		if _, ok := result[key]; !ok {
			return nil, fmt.Errorf("unknown model option %s", key)
		}
		if value != "" {
			result[key] = value
		}
	}
	for _, item := range []struct {
		key       string
		low, high int
	}{{"max_side", 256, 2048}, {"context_padding", 64, 1024}, {"mask_dilation", 0, 32}} {
		value, err := strconv.Atoi(result[item.key])
		if err != nil || value < item.low || value > item.high {
			return nil, fmt.Errorf("%s must be an integer between %d and %d", item.key, item.low, item.high)
		}
		result[item.key] = strconv.Itoa(value)
	}
	if result["seam_blend"] != "none" && result["seam_blend"] != "poisson" {
		return nil, errors.New("seam_blend must be none or poisson")
	}
	if result["context_mode"] != "full" && result["context_mode"] != "crop" {
		return nil, errors.New("context_mode must be full or crop")
	}
	return result, nil
}

func modelKey(url string, imageData, mask, protection []byte, options map[string]string) [32]byte {
	hash := sha256.New()
	// The worker rejects any artifact that does not match this pinned model.
	config, _ := json.Marshal(options)
	for _, part := range [][]byte{[]byte(url), []byte("lama:344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9:area-mask-v3"), config, imageData, mask, protection} {
		_ = binary.Write(hash, binary.LittleEndian, uint64(len(part)))
		_, _ = hash.Write(part)
	}
	var key [32]byte
	copy(key[:], hash.Sum(nil))
	return key
}

func orientedImageDimensions(data []byte) (int, int, error) {
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return 0, 0, err
	}
	if config.Width <= 0 || config.Height <= 0 || int64(config.Width)*int64(config.Height) > maxImagePixels {
		return 0, 0, errors.New("unsupported image dimensions")
	}
	if format == "jpeg" {
		if tags, _ := exif.Decode(bytes.NewReader(data)); tags != nil {
			if tag, err := tags.Get(exif.Orientation); err == nil {
				if value, err := tag.Int(0); err == nil && value >= 5 && value <= 8 {
					return config.Height, config.Width, nil
				}
			}
		}
	}
	return config.Width, config.Height, nil
}

func (s *Server) callBackgroundWorker(ctx context.Context, imageData, mask, protection []byte, values map[string]string) (modelResult, error) {
	options, err := modelOptions(values)
	if err != nil {
		return modelResult{}, fmt.Errorf("%w: %s", errWorkerRejected, err)
	}
	width, height, err := orientedImageDimensions(imageData)
	if err != nil {
		return modelResult{}, fmt.Errorf("%w: invalid model source image", errWorkerRejected)
	}
	for _, data := range [][]byte{mask, protection} {
		if len(data) == 0 {
			continue
		}
		config, err := png.DecodeConfig(bytes.NewReader(data))
		if err != nil || config.Width != width || config.Height != height {
			return modelResult{}, fmt.Errorf("%w: model masks must be PNGs matching the oriented source dimensions", errWorkerRejected)
		}
	}
	if len(mask) == 0 {
		return modelResult{}, fmt.Errorf("%w: model mask is required", errWorkerRejected)
	}
	if s.backgroundWorkerURL == "" {
		return modelResult{}, errBackgroundUnavailable
	}
	key := modelKey(s.backgroundWorkerURL, imageData, mask, protection, options)
	if result, ok := s.backgroundCache.get(key); ok {
		result.headers = result.headers.Clone()
		result.headers.Set("X-Inpaint-Cache", "hit")
		return result, nil
	}
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	for _, item := range []struct {
		name string
		data []byte
	}{{"image", imageData}, {"mask", mask}, {"protect_mask", protection}} {
		if len(item.data) == 0 {
			continue
		}
		part, err := form.CreateFormFile(item.name, item.name+".png")
		if err != nil {
			return modelResult{}, err
		}
		_, _ = part.Write(item.data)
	}
	for key, value := range options {
		if err := form.WriteField(key, value); err != nil {
			return modelResult{}, err
		}
	}
	if err := form.Close(); err != nil {
		return modelResult{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.backgroundWorkerURL+"/v1/inpaint", &body)
	if err != nil {
		return modelResult{}, errBackgroundUnavailable
	}
	request.Header.Set("Content-Type", form.FormDataContentType())
	response, err := s.backgroundClient.Do(request)
	if err != nil {
		if ctx.Err() != nil {
			return modelResult{}, ctx.Err()
		}
		return modelResult{}, errBackgroundUnavailable
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		if response.StatusCode == 422 {
			data, _ := io.ReadAll(io.LimitReader(response.Body, 8192))
			var detail struct {
				Detail string `json:"detail"`
			}
			_ = json.Unmarshal(data, &detail)
			return modelResult{}, fmt.Errorf("%w: %s", errWorkerRejected, safeWorkerMessage(detail.Detail))
		}
		return modelResult{}, errBackgroundUnavailable
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, s.maxUploadBytes+1))
	if err != nil || int64(len(data)) > s.maxUploadBytes {
		return modelResult{}, errBackgroundUnavailable
	}
	config, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil || config.Width != width || config.Height != height {
		return modelResult{}, errBackgroundUnavailable
	}
	if _, err := png.Decode(bytes.NewReader(data)); err != nil {
		return modelResult{}, errBackgroundUnavailable
	}
	result := modelResult{data: data, headers: make(http.Header)}
	for _, name := range modelHeaderNames {
		if value := response.Header.Get(name); value != "" {
			result.headers.Set(name, value)
		}
	}
	s.backgroundCache.put(key, result)
	result.headers = result.headers.Clone()
	result.headers.Set("X-Inpaint-Cache", "miss")
	return result, nil
}

func (s *Server) modelInpaintHTTP(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 3*s.maxUploadBytes)
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		writeError(w, 400, "invalid_model_input", "expected bounded multipart image and masks")
		return
	}
	defer r.MultipartForm.RemoveAll()
	read := func(name string, required bool) ([]byte, error) {
		file, _, err := r.FormFile(name)
		if errors.Is(err, http.ErrMissingFile) && !required {
			return nil, nil
		}
		if err != nil {
			return nil, errors.New(name + " is required")
		}
		defer file.Close()
		data, err := io.ReadAll(io.LimitReader(file, s.maxUploadBytes+1))
		if err != nil || int64(len(data)) > s.maxUploadBytes {
			return nil, errors.New(name + " is too large")
		}
		return data, nil
	}
	imageData, err := read("image", true)
	if err != nil {
		writeError(w, 400, "invalid_model_input", err.Error())
		return
	}
	mask, err := read("mask", true)
	if err != nil {
		writeError(w, 400, "invalid_model_input", err.Error())
		return
	}
	protection, err := read("protect_mask", false)
	if err != nil {
		writeError(w, 400, "invalid_model_input", err.Error())
		return
	}
	values := make(map[string]string)
	for key, items := range r.MultipartForm.Value {
		if len(items) != 1 {
			writeError(w, 422, "invalid_model_input", "model options must occur once")
			return
		}
		values[key] = items[0]
	}
	ctx, cancel := context.WithTimeout(r.Context(), s.renderTimeout)
	defer cancel()
	select {
	case s.renderSlots <- struct{}{}:
		defer func() { <-s.renderSlots }()
	case <-ctx.Done():
		writeRenderError(w, ctx.Err())
		return
	}
	result, err := s.callBackgroundWorker(ctx, imageData, mask, protection, values)
	if err != nil {
		writeRenderError(w, err)
		return
	}
	for name, values := range result.headers {
		w.Header()[name] = values
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(result.data)
}

func operationModelOptions(op Operation) map[string]string {
	values := make(map[string]string)
	for _, item := range [][2]string{{"maxSide", "max_side"}, {"contextMode", "context_mode"}, {"contextPadding", "context_padding"}, {"maskDilation", "mask_dilation"}, {"seamBlend", "seam_blend"}} {
		if value, ok := op.Params[item[0]]; ok {
			values[item[1]] = strings.TrimSpace(fmt.Sprint(value))
		}
	}
	return values
}
