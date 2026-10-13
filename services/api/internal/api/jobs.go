package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
)

type renderJob struct {
	ID         string          `json:"id"`
	SessionID  string          `json:"sessionId"`
	Status     string          `json:"status"`
	Progress   float64         `json:"progress"`
	Stage      string          `json:"stage"`
	CreatedAt  time.Time       `json:"createdAt"`
	FinishedAt *time.Time      `json:"finishedAt,omitempty"`
	ResultURL  string          `json:"resultUrl,omitempty"`
	Metadata   *renderMetadata `json:"metadata,omitempty"`
	Error      *jobError       `json:"error,omitempty"`
	cancel     context.CancelFunc
	output     string
}

type jobError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type jobRequest struct {
	SessionID string `json:"sessionId"`
	renderRequest
}

func (s *Server) startJob(sessionID string, req renderRequest) (renderJob, error) {
	sess, ok := s.getSession(sessionID)
	if !ok || sess.Image == nil {
		return renderJob{}, errors.New("session image not found")
	}
	graph, err := prepareGraph(req, sess)
	if err != nil {
		return renderJob{}, fmt.Errorf("%w: %s", errWorkerRejected, err)
	}
	if _, _, err := renderFormat(req, sess.Image); err != nil {
		return renderJob{}, fmt.Errorf("%w: %s", errWorkerRejected, err)
	}
	req.EditGraph = &graph
	req.Operation = nil
	id, err := newID()
	if err != nil {
		return renderJob{}, errors.New("job id generation failed")
	}
	ctx, cancel := context.WithCancel(s.ctx)
	job := &renderJob{ID: id, SessionID: sessionID, Status: "queued", Stage: "queued", CreatedAt: s.now(), cancel: cancel}
	s.mu.Lock()
	active := 0
	for _, j := range s.jobs {
		if j.Status == "queued" || j.Status == "running" {
			active++
		}
	}
	if active >= 64 {
		s.mu.Unlock()
		cancel()
		return renderJob{}, errors.New("render queue is full")
	}
	// Bound the temporary status/result cache. Completed jobs are disposable.
	if len(s.jobs) >= 256 {
		for key, j := range s.jobs {
			if j.Status != "queued" && j.Status != "running" {
				_ = os.Remove(j.output)
				delete(s.jobs, key)
				break
			}
		}
	}
	s.jobs[id] = job
	s.wg.Add(1)
	copy := *job
	s.mu.Unlock()
	go func() {
		defer s.wg.Done()
		defer cancel()
		output, err := s.runRender(ctx, sessionID, req, func(progress float64, stage string) {
			s.mu.Lock()
			if job.Status != "cancelled" {
				job.Status = "running"
				job.Progress = progress
				job.Stage = stage
			}
			s.mu.Unlock()
		})
		resultPath := ""
		if err == nil {
			file, e := os.CreateTemp(s.dataDir, "job-result-*")
			if e != nil {
				err = errors.New("job result storage failed")
			} else {
				resultPath = file.Name()
				_, e = file.Write(output.Data)
				closeErr := file.Close()
				if e != nil || closeErr != nil {
					err = errors.New("job result storage failed")
					_ = os.Remove(resultPath)
					resultPath = ""
				}
			}
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		now := s.now()
		job.FinishedAt = &now
		if job.Status == "cancelled" || errors.Is(err, context.Canceled) {
			job.Status = "cancelled"
			job.Stage = "cancelled"
			_ = os.Remove(resultPath)
			return
		}
		if err != nil {
			job.Status = "failed"
			job.Stage = "failed"
			job.Error = &jobError{Code: "render_failed", Message: "render could not be completed"}
			if errors.Is(err, errWorkerRejected) {
				job.Error = &jobError{Code: "invalid_operation", Message: err.Error()}
			}
			if errors.Is(err, context.DeadlineExceeded) {
				job.Error = &jobError{Code: "render_timeout", Message: "render exceeded the configured time limit"}
			}
			return
		}
		job.Status = "succeeded"
		job.Stage = "done"
		job.Progress = 1
		job.output = resultPath
		job.Metadata = &output.Metadata
		job.ResultURL = "/api/v1/jobs/" + job.ID + "/result"
	}()
	return copy, nil
}

func (s *Server) createJobHTTP(c *gin.Context) {
	var req jobRequest
	if json.NewDecoder(http.MaxBytesReader(c.Writer, c.Request.Body, 8<<20)).Decode(&req) != nil {
		writeError(c.Writer, 400, "invalid_job", "expected render job JSON")
		return
	}
	sessionID := c.Param("id")
	if sessionID == "" {
		sessionID = req.SessionID
	}
	job, err := s.startJob(sessionID, req.renderRequest)
	if err != nil {
		if errors.Is(err, errWorkerRejected) {
			writeRenderError(c.Writer, err)
		} else {
			writeError(c.Writer, 400, "invalid_job", err.Error())
		}
		return
	}
	c.JSON(http.StatusAccepted, job)
}

func (s *Server) getJobHTTP(c *gin.Context) {
	s.mu.RLock()
	job, ok := s.jobs[c.Param("jobID")]
	var snapshot renderJob
	if ok {
		snapshot = *job
	}
	s.mu.RUnlock()
	if !ok {
		notFound(c.Writer)
		return
	}
	c.JSON(200, snapshot)
}

func (s *Server) cancelJobHTTP(c *gin.Context) {
	s.mu.Lock()
	job, ok := s.jobs[c.Param("jobID")]
	if !ok {
		s.mu.Unlock()
		notFound(c.Writer)
		return
	}
	if job.Status == "queued" || job.Status == "running" {
		job.Status = "cancelled"
		job.Stage = "cancelled"
		job.cancel()
	}
	snapshot := *job
	s.mu.Unlock()
	c.JSON(200, snapshot)
}

func (s *Server) jobResultHTTP(c *gin.Context) {
	s.mu.RLock()
	job, ok := s.jobs[c.Param("jobID")]
	if !ok {
		s.mu.RUnlock()
		notFound(c.Writer)
		return
	}
	if job.Status != "succeeded" {
		s.mu.RUnlock()
		writeError(c.Writer, 409, "result_unavailable", "render has not succeeded")
		return
	}
	data, err := os.ReadFile(job.output)
	format := job.Metadata.Format
	s.mu.RUnlock()
	if err != nil {
		notFound(c.Writer)
		return
	}
	c.Header("Cache-Control", "no-store")
	c.Data(200, "image/"+format, data)
}
