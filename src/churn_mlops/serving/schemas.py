"""Contratos de entrada/saída da API de predição."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Customer(BaseModel):
    gender: str = Field(examples=["Female"])
    SeniorCitizen: str = Field(examples=["No"], description="Yes/No")
    Partner: str = Field(examples=["Yes"])
    Dependents: str = Field(examples=["No"])
    tenure: float = Field(ge=0, examples=[12])
    PhoneService: str = Field(examples=["Yes"])
    MultipleLines: str = Field(examples=["No"])
    InternetService: str = Field(examples=["Fiber optic"])
    OnlineSecurity: str = Field(examples=["No"])
    OnlineBackup: str = Field(examples=["No"])
    DeviceProtection: str = Field(examples=["No"])
    TechSupport: str = Field(examples=["No"])
    StreamingTV: str = Field(examples=["Yes"])
    StreamingMovies: str = Field(examples=["No"])
    Contract: str = Field(examples=["Month-to-month"])
    PaperlessBilling: str = Field(examples=["Yes"])
    PaymentMethod: str = Field(examples=["Electronic check"])
    MonthlyCharges: float = Field(ge=0, examples=[89.9])
    TotalCharges: float = Field(ge=0, examples=[1078.8])


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=10_000)


class Prediction(BaseModel):
    churn_probability: float
    churn_predicted: bool


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    threshold: float


class ModelInfo(BaseModel):
    model_version: str
    trained_at: str
    threshold: float
    features: list[str]
    metrics: dict
