"""Request and response bodies for the MiBici risk prediction API."""

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StationFeatures(BaseModel):
    """Features of a single station at minute 50 of hour t."""

    model_config = ConfigDict(extra="forbid")

    station_id: int | None = None
    capacity: int = Field(..., gt=0)
    latitude: float = Field(..., ge=20.4, le=21.0)
    longitude: float = Field(..., ge=-103.6, le=-103.1)
    hour_sin: float = Field(..., ge=-1.0, le=1.0)
    hour_cos: float = Field(..., ge=-1.0, le=1.0)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: bool
    is_holiday: bool
    neighbor_net_flow: float
    net_flow_lag1: float
    net_flow_lag2: float
    net_flow_lag24: float
    arrivals_lag1: float = Field(..., ge=0.0)
    arrivals_lag2: float = Field(..., ge=0.0)
    arrivals_lag3: float = Field(..., ge=0.0)
    departures_lag1: float = Field(..., ge=0.0)
    departures_lag2: float = Field(..., ge=0.0)
    departures_lag3: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def validate_weekend_consistency(self) -> "StationFeatures":
        """Enforce temporal consistency between day of week and weekend flag."""
        expected_weekend = self.day_of_week >= 5
        if self.is_weekend != expected_weekend:
            raise ValueError(
                f"is_weekend={self.is_weekend} is inconsistent with day_of_week={self.day_of_week} "
                f"(expected {expected_weekend})"
            )
        return self


class PredictRequest(BaseModel):
    """Batch of station features to classify in one call."""

    model_config = ConfigDict(extra="forbid")

    instances: list[StationFeatures] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    """Predicted risk class and class probabilities for a single station."""

    model_config = ConfigDict(extra="forbid")

    station_id: int | None = None
    label: str
    label_id: int = Field(..., ge=0, le=2)
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def validate_probabilities_sum(self) -> "Prediction":
        """Enforce that predicted class probabilities sum to approximately 1.0."""
        total = sum(self.probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Probabilities must sum to ~1.0, got {total}")
        return self


class PredictResponse(BaseModel):
    """Risk predictions accompanied by the producing model version and decision thresholds."""

    model_config = ConfigDict(extra="forbid")

    model_name: str | None = None
    model_version: str | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    """Registry version and lineage metadata of the served model."""

    model_config = ConfigDict(extra="forbid")

    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    source_run_id: str | None = None
    git_commit: str | None = None
    data_version: str | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    exported_at: str | None = None
