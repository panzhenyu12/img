.DEFAULT_GOAL := help

API_DIR := services/api
WEB_DIR := apps/web
API_ADDR ?= http://127.0.0.1:8080
WEB_ADDR ?= http://127.0.0.1:5173

.PHONY: help setup dev api web agent web-install native wasm api-build build test check fmt \
	rust-test api-test web-build e2e health stop clean models quality quality-face quality-body quality-clothing lint

help:
	@echo "make dev          启动 Go API 和 Vite Web 开发服务器"
	@echo "make setup        安装依赖、WASM 工具和可选识别模型"
	@echo "make api          只启动 Go API"
	@echo "make web          只启动 Vite Web"
	@echo "make test         运行 Rust、Go 和 Web 检查"
	@echo "make e2e          验证 WASM 预览、画笔、历史和 native 导出"
	@echo "make quality      在真实人像/身形/衣物照片上生成对比和失败报告"
	@echo "make agent        启动可选 DeepAgent (8090)"
	@echo "make check        运行测试、vet 和生产构建"
	@echo "make health       检查本地 Go API"
	@echo "make stop         停止 8080/5173 本地开发进程"
	@echo "make clean        清理构建产物和缓存"

setup: web-install
	rustup target add wasm32-unknown-unknown
	cargo install wasm-bindgen-cli --version 0.2.129 --locked
	cd $(WEB_DIR) && npx playwright install chromium
	$(MAKE) models

dev: native wasm api-build
	python3 scripts/dev.py start

api: native
	cd $(API_DIR) && go run ./cmd/server

web: wasm
	cd $(WEB_DIR) && npm run dev -- --host 127.0.0.1

agent:
	cd services/agent && uv run uvicorn main:app --host 127.0.0.1 --port 8090

native:
	cargo build --release --manifest-path core/image-core/Cargo.toml --bin image-render

wasm:
	cargo build --release --target wasm32-unknown-unknown --features wasm --lib --manifest-path core/image-core/Cargo.toml
	wasm-bindgen core/image-core/target/wasm32-unknown-unknown/release/image_core.wasm --target web --out-dir $(WEB_DIR)/public/wasm

api-build: catalog
	mkdir -p .run
	cd $(API_DIR) && go build -o ../../.run/api ./cmd/server

models:
	python3 scripts/setup_models.py

.PHONY: catalog
catalog:
	node scripts/generate_catalog.mjs

web-install:
	cd $(WEB_DIR) && npm ci --no-audit --no-fund

build: native wasm rust-test api-test web-build

test: native wasm rust-test api-test web-build

check: test lint

lint:
	cargo clippy --all-targets --manifest-path core/image-core/Cargo.toml -- -D warnings

# Real-photo diagnostics retain FAIL/UNVERIFIED. A successful render alone
# does not establish a natural contour or professional retouching quality.
quality: dev
	python3 scripts/check_quality.py

quality-face: dev
	node scripts/check_portrait_quality.mjs --batch

quality-body: dev
	node scripts/check_body_quality.mjs

quality-clothing: dev
	node scripts/check_clothing_quality.mjs

e2e: dev
	cd $(WEB_DIR) && npx playwright test

fmt:
	cargo fmt --manifest-path core/image-core/Cargo.toml
	cd $(API_DIR) && gofmt -w $$(find . -name '*.go' -type f)

rust-test:
	cargo test --manifest-path core/image-core/Cargo.toml

api-test: catalog
	cd $(API_DIR) && go test ./... && go vet ./...

web-build:
	cd $(WEB_DIR) && npm run build

health:
	curl -fsS $(API_ADDR)/healthz
	@echo

stop:
	python3 scripts/dev.py stop

clean: stop
	python3 scripts/dev.py clean
