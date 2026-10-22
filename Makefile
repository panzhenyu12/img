.DEFAULT_GOAL := help

API_DIR := services/api
WEB_DIR := apps/web
API_ADDR ?= http://127.0.0.1:8080
WEB_ADDR ?= http://127.0.0.1:5173
QUALITY_PYTHON ?= services/background-worker/.venv/bin/python
SEMANTIC_PYTHON ?= $(QUALITY_PYTHON)

.PHONY: help setup dev api web agent web-install native wasm api-build build test check fmt \
	rust-test api-test web-build e2e health stop clean models quality quality-face quality-body quality-clothing quality-semantic quality-semantic-local lint

.PHONY: background-setup background-models background background-test client-models quality-skin quality-film body-setup body-models rust-photo-test
.PHONY: quality-jaw-curve quality-jaw-bilateral quality-jaw-retained quality-jaw-joint

help:
	@echo "make dev          启动 Go API 和 Vite Web 开发服务器"
	@echo "make setup        安装依赖、WASM 工具和可选识别模型"
	@echo "make api          只启动 Go API"
	@echo "make web          只启动 Vite Web"
	@echo "make test         运行 Rust、Go 和 Web 检查"
	@echo "make e2e          验证 WASM 预览、画笔、历史和 native 导出"
	@echo "make quality      在真实人像/身形/衣物照片上生成对比和失败报告"
	@echo "make quality-film 在六张照片上回放胶片响应，保留哈希和首次/热运行时间"
	@echo "make quality-semantic 运行语义形变隔离测试，不构建 Web/Go，不读取本地照片"
	@echo "make quality-semantic-local 另验本地照片输入合同；照片及数组缺失时跳过"
	@echo "make quality-jaw-curve 只验下颌曲线与强度传递算法，无照片/系统构建"
	@echo "make quality-jaw-bilateral 只验双侧下颌目标、导出与独立指标，无系统构建"
	@echo "make background   启动可选 LaMa 本地去人模型 (8091)"
	@echo "make background-models 安装可选模型环境并下载校验权重"
	@echo "make client-models 下载校验浏览器去人 ONNX 模型"
	@echo "make body-models  导出并校验可选人体像素分区模型"
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

background-setup:
	@test -x services/background-worker/.venv/bin/python || uv venv --python 3.11 services/background-worker/.venv
	uv pip install --python services/background-worker/.venv/bin/python -r services/background-worker/requirements.txt

background-models: background-setup
	services/background-worker/.venv/bin/python services/background-worker/scripts/download_models.py lama-torchscript

client-models:
	python3 scripts/setup_client_inpaint.py

body-setup:
	@test -x .models/body-parser/.venv/bin/python || uv venv --python 3.11 .models/body-parser/.venv
	uv pip install --python .models/body-parser/.venv/bin/python -r scripts/body_parser_requirements.txt

body-models: body-setup
	.models/body-parser/.venv/bin/python scripts/export_body_parser.py

background:
	cd services/background-worker && .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8091

background-test:
	services/background-worker/.venv/bin/python -m pytest services/background-worker/tests -q

quality-skin:
	node scripts/check_portrait_quality.mjs --features

# Install experiments/semantic-deformation-20261022/requirements.txt into the
# chosen SEMANTIC_PYTHON environment. No Web/Go/native-image build is needed.
quality-semantic:
	cd experiments/semantic-deformation-20261022 && $(abspath $(SEMANTIC_PYTHON)) -m unittest test_reduced_field.py test_reduced_constrained.py test_focused_correction.py test_prepare_hair_inputs.py test_resampling.py test_regional_input.py test_source_regions.py test_jaw_input.py test_portable_field.py test_portable_jaw.py test_portable_result_gate.py test_resolution_field.py test_jaw_collocation.py test_source_boundary_support.py
	cargo test --offline --manifest-path experiments/semantic-deformation-20261022/rust-controls/Cargo.toml

# This optional target reads ignored source arrays and photo bytes (hash only).
quality-semantic-local:
	cd experiments/semantic-deformation-20261022 && $(abspath $(SEMANTIC_PYTHON)) -m unittest test_input_contract.py test_controller_adapter.py test_jaw_local.py

quality-jaw-curve:
	cd experiments/semantic-deformation-20261022 && $(abspath $(SEMANTIC_PYTHON)) -m unittest test_jaw_curve_qp.py test_jaw_qp_precision.py test_jaw_target_profile.py test_export_jaw_curve_profile.py test_jaw_collocation.py test_source_boundary_soft_targets.py test_boundary_curve_qp.py test_export_boundary_curve.py test_audit_interface_chord_field.py

quality-jaw-bilateral:
	cd experiments/semantic-deformation-20261022 && $(abspath $(SEMANTIC_PYTHON)) -m unittest test_jaw_bilateral.py test_jaw_curve_qp.py test_jaw_qp_precision.py test_jaw_target_profile.py test_export_jaw_curve_profile.py test_audit_jaw_bilateral_field.py

quality-jaw-retained:
	cd experiments/semantic-deformation-20261022 && $(abspath $(SEMANTIC_PYTHON)) -m unittest test_jaw_retained_bend.py test_audit_jaw_retained_bend.py

quality-jaw-joint:
	cd experiments/semantic-deformation-20261022 && $(abspath $(SEMANTIC_PYTHON)) -m unittest test_joint_jaw_boundary_field.py test_rank_one_inverse_contract.py test_interval_polynomial_cones.py test_inverse_field_linearization.py test_joint_inverse_feedback.py

# Requires Pillow/numpy; override QUALITY_PYTHON to use another environment.
quality-film: native
	cargo build --release --manifest-path core/image-core/Cargo.toml --example graph_quality_bench
	$(QUALITY_PYTHON) scripts/review_film_response.py --out test/results/film-response-current

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

# Photo-derived fixtures are local-only and must never be compiled into Git.
rust-photo-test:
	cargo test --release --manifest-path core/image-core/Cargo.toml local_photo_ -- --ignored

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
