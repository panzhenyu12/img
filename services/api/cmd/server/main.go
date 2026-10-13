package main

import (
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/modao/img/internal/api"
	"github.com/spf13/viper"
)

func main() {
	configure()
	dataDir := viper.GetString("server.data_dir")
	if dataDir == "" {
		dataDir = filepath.Join(os.TempDir(), "image-editor-api")
	}
	addr := viper.GetString("server.addr")
	sessionTTL := viper.GetDuration("server.session_ttl")
	maxUploadBytes := viper.GetInt64("server.max_upload_bytes")

	server, err := api.NewServer(api.Config{
		DataDir:           dataDir,
		SessionTTL:        sessionTTL,
		MaxUploadBytes:    maxUploadBytes,
		RenderBinary:      viper.GetString("render.binary"),
		RenderTimeout:     viper.GetDuration("render.timeout"),
		RenderConcurrency: viper.GetInt("render.concurrency"),
	})
	if err != nil {
		log.Fatalf("create api server: %v", err)
	}
	defer server.Close()

	httpServer := &http.Server{
		Addr:              addr,
		Handler:           server.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      viper.GetDuration("render.timeout") + 10*time.Second,
		IdleTimeout:       120 * time.Second,
	}
	log.Printf("image editor API listening on %s (data: %s)", addr, dataDir)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func configure() {
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("/etc/image-editor")
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	viper.AutomaticEnv()
	// Keep the Sprint 0 variable names documented by README stable while also
	// exposing namespaced keys for config.yaml.
	_ = viper.BindEnv("server.addr", "IMG_API_ADDR")
	_ = viper.BindEnv("server.data_dir", "IMG_API_DATA_DIR")
	_ = viper.BindEnv("server.session_ttl", "IMG_API_SESSION_TTL")
	_ = viper.BindEnv("server.max_upload_bytes", "IMG_API_MAX_UPLOAD_BYTES")
	_ = viper.BindEnv("render.binary", "IMG_RENDER_BINARY")
	_ = viper.BindEnv("render.timeout", "IMG_RENDER_TIMEOUT")
	_ = viper.BindEnv("render.concurrency", "IMG_RENDER_CONCURRENCY")
	viper.SetDefault("render.binary", "")
	viper.SetDefault("render.timeout", 2*time.Minute)
	viper.SetDefault("render.concurrency", 2)
	viper.SetDefault("server.addr", ":8080")
	viper.SetDefault("server.data_dir", "")
	viper.SetDefault("server.session_ttl", 24*time.Hour)
	viper.SetDefault("server.max_upload_bytes", int64(100<<20))
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			log.Printf("read config: %v", err)
		}
	}
}
