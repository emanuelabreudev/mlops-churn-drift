# Pipeline de MLOps — churn com monitoramento de drift
# Fluxo completo: make all  (setup -> dados -> eda -> treino -> drift -> simulação)

.PHONY: setup data preprocess eda train drift simulate ablation all api dashboard mlflow-ui test lint docker-build docker-up clean

setup:
	uv sync --group dev

data:
	uv run python -m churn_mlops.data.download

preprocess: data
	uv run python -m churn_mlops.data.preprocess

eda: preprocess
	uv run python -m churn_mlops.eda

train: preprocess
	uv run python -m churn_mlops.models.train

drift: train
	uv run python -m churn_mlops.drift.scenario

simulate: drift
	uv run python -m churn_mlops.pipeline.simulate

ablation: train
	uv run python -m churn_mlops.pipeline.ablation

all: setup eda simulate

api:
	uv run uvicorn churn_mlops.serving.app:app --host 0.0.0.0 --port 8000

dashboard:
	uv run streamlit run dashboard/app.py

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	uv run pytest

lint:
	uv run ruff check src tests dashboard

docker-build:
	docker compose build

docker-up:
	docker compose up -d

clean:
	rm -rf data/processed reports artifacts mlflow.db mlruns
