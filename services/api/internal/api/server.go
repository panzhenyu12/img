// Package api implements the temporary, stateless API used by the first web
// editor milestone. It intentionally keeps sessions in memory and stores image
// bytes in a temporary directory. The storage interface can be replaced by an
// object store when project persistence is introduced.
package api

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/modao/img/internal/ai"

	"github.com/gin-gonic/gin"
	"github.com/rwcarlsen/goexif/exif"
	_ "golang.org/x/image/webp"
)

const (
	defaultMaxUploadBytes = 100 << 20
	maxImagePixels        = 100_000_000
)

// Config controls the temporary API server.
type Config struct {
	DataDir             string
	SessionTTL          time.Duration
	MaxUploadBytes      int64
	Now                 func() time.Time
	RenderBinary        string
	RenderTimeout       time.Duration
	RenderConcurrency   int
	BackgroundWorkerURL string
}

// Server is the in-memory session API. A Server is safe for concurrent use.
type Server struct {
	dataDir        string
	sessionTTL     time.Duration
	maxUploadBytes int64
	now            func() time.Time

	mu                  sync.RWMutex
	sessions            map[string]*session
	renderBinary        string
	renderTimeout       time.Duration
	renderSlots         chan struct{}
	jobs                map[string]*renderJob
	ctx                 context.Context
	cancel              context.CancelFunc
	wg                  sync.WaitGroup
	backgroundWorkerURL string
	backgroundClient    *http.Client
	backgroundCache     *modelCache
}

type session struct {
	ID             string    `json:"id"`
	CreatedAt      time.Time `json:"createdAt"`
	ExpiresAt      time.Time `json:"expiresAt"`
	Image          *asset    `json:"image,omitempty"`
	Revision       string    `json:"revision"`
	EditGraph      EditGraph `json:"editGraph"`
	revisionNumber uint64
	history        []EditGraph
	proposals      map[string]ai.EditProposal
	assetPaths     []string
}

type asset struct {
	ID         string    `json:"id"`
	Filename   string    `json:"filename"`
	MimeType   string    `json:"mimeType"`
	Size       int64     `json:"size"`
	Width      int       `json:"width"`
	Height     int       `json:"height"`
	UploadedAt time.Time `json:"uploadedAt"`
	Path       string    `json:"-"`
}

// Operation is shared by manual editing, AI and MCP.
type Operation = ai.Operation

type EditGraph struct {
	Version       string      `json:"version,omitempty"`
	SourceAssetID string      `json:"sourceAssetId,omitempty"`
	Operations    []Operation `json:"operations,omitempty"`
	// Nodes is accepted as an alias while the schema is being migrated from the
	// early prototype format.
	Nodes []Operation `json:"nodes,omitempty"`
}

type renderRequest struct {
	EditGraph *EditGraph `json:"editGraph,omitempty"`
	Operation *Operation `json:"operation,omitempty"`
	Format    string     `json:"format,omitempty"`
	Quality   int        `json:"quality,omitempty"`
	// Committing a graph or proposal must validate even disabled operations.
	forceCore bool
}

// NewServer creates a temporary API server and its data directory.
func NewServer(cfg Config) (*Server, error) {
	if cfg.DataDir == "" {
		cfg.DataDir = filepath.Join(os.TempDir(), "image-editor-api")
	}
	if cfg.SessionTTL <= 0 {
		cfg.SessionTTL = 24 * time.Hour
	}
	if cfg.MaxUploadBytes <= 0 {
		cfg.MaxUploadBytes = defaultMaxUploadBytes
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if err := os.MkdirAll(cfg.DataDir, 0o700); err != nil {
		return nil, fmt.Errorf("create data directory: %w", err)
	}
	if cfg.RenderTimeout <= 0 {
		cfg.RenderTimeout = 2 * time.Minute
	}
	if cfg.RenderConcurrency <= 0 {
		cfg.RenderConcurrency = 2
	}
	if cfg.RenderBinary == "" {
		cfg.RenderBinary = findRenderBinary()
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &Server{
		dataDir:        cfg.DataDir,
		sessionTTL:     cfg.SessionTTL,
		maxUploadBytes: cfg.MaxUploadBytes,
		now:            cfg.Now,
		sessions:       make(map[string]*session),
		renderBinary:   cfg.RenderBinary, renderTimeout: cfg.RenderTimeout, renderSlots: make(chan struct{}, cfg.RenderConcurrency), jobs: make(map[string]*renderJob), ctx: ctx, cancel: cancel,
		backgroundWorkerURL: strings.TrimRight(cfg.BackgroundWorkerURL, "/"),
		backgroundClient:    &http.Client{Timeout: cfg.RenderTimeout},
		backgroundCache:     newModelCache(64 << 20),
	}, nil
}

// Close removes temporary files created by this server. It does not remove a
// caller-provided data directory itself, which makes local inspection easier.
func (s *Server) Close() {
	s.cancel()
	s.wg.Wait()
	s.mu.Lock()
	defer s.mu.Unlock()
	for id := range s.sessions {
		s.deleteSessionLocked(id)
	}
	for _, job := range s.jobs {
		if job.output != "" {
			_ = os.Remove(job.output)
		}
	}
}

// Handler returns the HTTP handler for embedding or testing.
func (s *Server) Handler() http.Handler {
	engine := gin.New()
	engine.Use(gin.Recovery(), ginCORSMiddleware())
	engine.GET("/healthz", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "ok"}) })
	engine.POST("/api/inpaint/model", func(c *gin.Context) { s.modelInpaintHTTP(c.Writer, c.Request) })
	engine.POST("/api/v1/sessions", func(c *gin.Context) { s.createSession(c.Writer) })
	engine.GET("/api/v1/sessions/:id", func(c *gin.Context) { s.session(c.Writer, c.Request, c.Param("id")) })
	engine.DELETE("/api/v1/sessions/:id", func(c *gin.Context) { s.session(c.Writer, c.Request, c.Param("id")) })
	for _, action := range []string{"assets", "upload", "image", "source", "preview", "render"} {
		engine.Any("/api/v1/sessions/:id/"+action, func(c *gin.Context) {
			parts := strings.Split(c.Request.URL.Path, "/")
			s.sessionAction(c.Writer, c.Request, c.Param("id"), parts[len(parts)-1])
		})
	}
	engine.GET("/api/v1/sessions/:id/proposals", s.listProposalsHTTP)
	engine.POST("/api/v1/sessions/:id/proposals", s.proposeHTTP)
	engine.POST("/api/v1/sessions/:id/proposals/:proposalID/apply", s.applyHTTP)
	engine.POST("/api/v1/sessions/:id/proposals/:proposalID/reject", s.rejectHTTP)
	engine.POST("/api/v1/sessions/:id/operations", s.operationsHTTP)
	engine.PUT("/api/v1/sessions/:id/graph", s.graphHTTP)
	engine.POST("/api/v1/sessions/:id/undo", s.undoHTTP)
	engine.POST("/api/v1/jobs", s.createJobHTTP)
	engine.POST("/api/v1/sessions/:id/jobs", s.createJobHTTP)
	engine.GET("/api/v1/jobs/:jobID", s.getJobHTTP)
	engine.GET("/api/v1/jobs/:jobID/result", s.jobResultHTTP)
	engine.POST("/api/v1/jobs/:jobID/cancel", s.cancelJobHTTP)
	engine.GET("/api/v1/tools", func(c *gin.Context) { c.JSON(200, gin.H{"tools": toolCatalog()}) })
	engine.GET("/api/v1/operations", func(c *gin.Context) { c.JSON(200, gin.H{"operations": operationCatalog()}) })
	engine.POST("/api/v1/tools/:name/call", s.toolHTTP)
	engine.Any("/mcp", gin.WrapH(s.MCPHandler()))
	return engine
}

func ginCORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Idempotency-Key, X-Filename, MCP-Protocol-Version, MCP-Session-Id, Last-Event-ID")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		if c.Request.Method == http.MethodOptions {
			c.Status(http.StatusNoContent)
			c.Abort()
			return
		}
		c.Next()
	}
}

func (s *Server) createSession(w http.ResponseWriter) {
	id, err := newID()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "id_generation_failed", err.Error())
		return
	}
	now := s.now()
	sess := &session{ID: id, CreatedAt: now, ExpiresAt: now.Add(s.sessionTTL), Revision: "0", proposals: make(map[string]ai.EditProposal), EditGraph: EditGraph{Version: "1.0", Operations: []Operation{}}}
	s.mu.Lock()
	s.sessions[id] = sess
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, sess)
}

func (s *Server) session(w http.ResponseWriter, r *http.Request, id string) {
	sess, ok := s.getSession(id)
	if !ok {
		notFound(w)
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, sess)
	case http.MethodDelete:
		s.deleteSession(id)
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w, http.MethodGet, http.MethodDelete)
	}
}

func (s *Server) sessionAction(w http.ResponseWriter, r *http.Request, id, action string) {
	if _, ok := s.getSession(id); !ok {
		notFound(w)
		return
	}
	switch action {
	case "assets", "upload", "image":
		if r.Method == http.MethodPost {
			s.upload(w, r, id)
			return
		}
		if r.Method == http.MethodGet {
			s.source(w, r, id)
			return
		}
		methodNotAllowed(w, http.MethodGet, http.MethodPost)
	case "preview", "render":
		if r.Method == http.MethodPost {
			s.render(w, r, id)
			return
		}
		if r.Method == http.MethodGet {
			s.source(w, r, id)
			return
		}
		methodNotAllowed(w, http.MethodGet, http.MethodPost)
	case "source":
		if r.Method == http.MethodGet {
			s.source(w, r, id)
			return
		}
		methodNotAllowed(w, http.MethodGet)
	default:
		notFound(w)
	}
}

func (s *Server) upload(w http.ResponseWriter, r *http.Request, sessionID string) {
	if r.ContentLength > s.maxUploadBytes && r.ContentLength >= 0 {
		writeError(w, http.StatusRequestEntityTooLarge, "upload_too_large", "the upload exceeds the configured limit")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadBytes+1)
	var (
		reader   io.Reader
		filename = "image"
		mimeType string
		closeFn  func() error
	)
	if strings.HasPrefix(strings.ToLower(r.Header.Get("Content-Type")), "multipart/") {
		file, header, err := r.FormFile("file")
		if errors.Is(err, http.ErrMissingBoundary) || err != nil {
			if err != nil {
				file, header, err = r.FormFile("image")
			}
		}
		if err != nil {
			writeError(w, http.StatusBadRequest, "missing_file", "multipart field 'file' is required")
			return
		}
		defer file.Close()
		defer r.MultipartForm.RemoveAll()
		reader = file
		filename = sanitizeFilename(header.Filename)
		mimeType = header.Header.Get("Content-Type")
	} else {
		reader = r.Body
		filename = sanitizeFilename(r.Header.Get("X-Filename"))
		if filename == "image" {
			filename = "image.bin"
		}
		mimeType = r.Header.Get("Content-Type")
		closeFn = r.Body.Close
	}
	if closeFn != nil {
		defer closeFn()
	}

	tmp, err := os.CreateTemp(s.dataDir, "upload-*")
	if err != nil {
		writeError(w, http.StatusInternalServerError, "storage_failed", "temporary storage is unavailable")
		return
	}
	defer tmp.Close()
	tmpPath := tmp.Name()
	defer func() { _ = os.Remove(tmpPath) }()
	n, err := io.Copy(tmp, io.LimitReader(reader, s.maxUploadBytes+1))
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeError(w, http.StatusRequestEntityTooLarge, "upload_too_large", "the upload exceeds the configured limit")
			return
		}
		writeError(w, http.StatusBadRequest, "upload_read_failed", err.Error())
		return
	}
	if n > s.maxUploadBytes {
		writeError(w, http.StatusRequestEntityTooLarge, "upload_too_large", "the upload exceeds the configured limit")
		return
	}
	if err := tmp.Close(); err != nil {
		writeError(w, http.StatusInternalServerError, "storage_failed", "temporary storage is unavailable")
		return
	}

	meta, err := inspectImage(tmpPath)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, "invalid_image", err.Error())
		return
	}
	if mimeType == "" {
		mimeType = meta.mimeType
	}
	// Trust the bytes over a client-provided MIME value.
	mimeType = meta.mimeType
	assetID, err := newID()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "id_generation_failed", err.Error())
		return
	}
	finalPath := filepath.Join(s.dataDir, sessionID+"-"+assetID+"."+meta.extension)
	if err := os.Rename(tmpPath, finalPath); err != nil {
		writeError(w, http.StatusInternalServerError, "storage_failed", "temporary storage is unavailable")
		return
	}
	now := s.now()
	a := &asset{ID: assetID, Filename: filename, MimeType: mimeType, Size: n, Width: meta.width, Height: meta.height, UploadedAt: now, Path: finalPath}
	s.mu.Lock()
	current, exists := s.sessions[sessionID]
	if !exists || !s.now().Before(current.ExpiresAt) {
		s.mu.Unlock()
		_ = os.Remove(finalPath)
		notFound(w)
		return
	}
	current.Image = a
	current.assetPaths = append(current.assetPaths, finalPath)
	current.revisionNumber++
	current.Revision = fmt.Sprint(current.revisionNumber)
	current.EditGraph = EditGraph{Version: "1.0", SourceAssetID: a.ID, Operations: []Operation{}}
	current.history = nil
	current.proposals = make(map[string]ai.EditProposal)
	s.mu.Unlock()
	writeJSON(w, http.StatusCreated, a)
}

func (s *Server) source(w http.ResponseWriter, _ *http.Request, sessionID string) {
	sess, ok := s.getSession(sessionID)
	if !ok || sess.Image == nil {
		writeError(w, http.StatusNotFound, "image_not_found", "upload an image before requesting a preview")
		return
	}
	f, err := os.Open(sess.Image.Path)
	if err != nil {
		writeError(w, http.StatusNotFound, "image_not_found", "the temporary image is no longer available")
		return
	}
	defer f.Close()
	w.Header().Set("Content-Type", sess.Image.MimeType)
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Asset-ID", sess.Image.ID)
	_, _ = io.Copy(w, f)
}

type imageMeta struct {
	width, height int
	mimeType      string
	extension     string
}

func inspectImage(path string) (imageMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return imageMeta{}, err
	}
	defer f.Close()
	config, format, err := image.DecodeConfig(f)
	if err != nil {
		return imageMeta{}, fmt.Errorf("decode image: %w", err)
	}
	if config.Width <= 0 || config.Height <= 0 || int64(config.Width)*int64(config.Height) > maxImagePixels {
		return imageMeta{}, fmt.Errorf("image dimensions are unsupported (%dx%d)", config.Width, config.Height)
	}
	meta := imageMeta{width: config.Width, height: config.Height}
	switch strings.ToLower(format) {
	case "jpeg":
		meta.mimeType, meta.extension = "image/jpeg", "jpg"
		// Browsers and the Rust decoder apply EXIF orientation before displaying
		// pixels. Only read metadata here so AI coordinates use the same bounds.
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return imageMeta{}, fmt.Errorf("read image metadata: %w", err)
		}
		if tags, _ := exif.Decode(f); tags != nil {
			if tag, err := tags.Get(exif.Orientation); err == nil {
				if orientation, err := tag.Int(0); err == nil && orientation >= 5 && orientation <= 8 {
					meta.width, meta.height = meta.height, meta.width
				}
			}
		}
	case "png":
		meta.mimeType, meta.extension = "image/png", "png"
	case "gif":
		meta.mimeType, meta.extension = "image/gif", "gif"
	case "webp":
		meta.mimeType, meta.extension = "image/webp", "webp"
	default:
		return imageMeta{}, fmt.Errorf("image format %q is not supported", format)
	}
	return meta, nil
}

func (s *Server) getSession(id string) (*session, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[id]
	if !ok {
		return nil, false
	}
	if !s.now().Before(sess.ExpiresAt) {
		s.deleteSessionLocked(id)
		return nil, false
	}
	copied := *sess
	if sess.Image != nil {
		a := *sess.Image
		copied.Image = &a
	}
	copied.EditGraph = cloneGraph(sess.EditGraph)
	return &copied, true
}

func (s *Server) deleteSession(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.deleteSessionLocked(id)
}

func (s *Server) deleteSessionLocked(id string) {
	if sess, ok := s.sessions[id]; ok {
		for _, path := range sess.assetPaths {
			_ = os.Remove(path)
		}
		delete(s.sessions, id)
	}
}

func cloneGraph(graph EditGraph) EditGraph {
	data, _ := json.Marshal(graph)
	var copied EditGraph
	_ = json.Unmarshal(data, &copied)
	return copied
}

func newID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func sanitizeFilename(name string) string {
	name = filepath.Base(strings.TrimSpace(name))
	if name == "." || name == "" || name == string(filepath.Separator) {
		return "image"
	}
	if len(name) > 200 {
		name = name[:200]
	}
	return name
}

func setCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Idempotency-Key, X-Filename, MCP-Protocol-Version, MCP-Session-Id, Last-Event-ID")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"code": code, "message": message}})
}

func notFound(w http.ResponseWriter) {
	writeError(w, http.StatusNotFound, "not_found", "resource not found")
}

func methodNotAllowed(w http.ResponseWriter, methods ...string) {
	w.Header().Set("Allow", strings.Join(methods, ", "))
	writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "method is not allowed")
}
