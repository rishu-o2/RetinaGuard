.PHONY: help install test test-backend test-ml test-frontend lint format run-backend run-frontend docker-up docker-down data-aptos data-verify data-idrid

help:
	@echo "RetinaGuard Automation Commands:"
	@echo "  make install        Install backend and ML dependencies"
	@echo "  make test           Run all tests across backend, ML, and frontend"
	@echo "  make test-backend   Run pytest on backend/"
	@echo "  make test-ml        Run pytest on ml/"
	@echo "  make test-frontend  Run node test runner on frontend/"
	@echo "  make lint           Run ruff linting"
	@echo "  make format         Run ruff formatting"
	@echo "  make data-aptos     Download and verify APTOS 2019 dataset"
	@echo "  make data-verify    Verify APTOS 2019 dataset integrity"
	@echo "  make data-idrid     Prepare and verify IDRiD external test dataset"
	@echo "  make run-backend    Start FastAPI backend server locally"
	@echo "  make run-frontend   Start React/Vite development server"
	@echo "  make docker-up      Start containerized services via docker-compose"
	@echo "  make docker-down    Stop containerized services"

install:
	pip install -e backend
	pip install -e ml

test-backend:
	python -m pytest backend/tests/

test-ml:
	python -m pytest ml/tests/

test-frontend:
	node --test frontend/tests/*.test.js

test: test-backend test-ml test-frontend

lint:
	ruff check backend/ ml/

format:
	ruff format backend/ ml/

data-aptos:
	python ml/scripts/download_aptos.py
	python ml/scripts/verify_aptos.py

data-verify:
	python ml/scripts/verify_aptos.py

data-idrid:
	python ml/scripts/prepare_idrid.py
	python ml/scripts/verify_idrid.py

run-backend:
	uvicorn retinaguard.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm run dev

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
