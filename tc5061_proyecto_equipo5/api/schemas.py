"""Request and response bodies of the P2 API (provisional: WP-2 adds the validations)."""

from pydantic import BaseModel


class StationFeatures(BaseModel):
    """One station at minute 50 of hour t, with the 18 features the model expects."""

    station_id: int | None = None
    capacity: int
    latitude: float
    longitude: float
    hour_sin: float
    hour_cos: float
    day_of_week: int
    is_weekend: bool
    is_holiday: bool
    neighbor_net_flow: float
    net_flow_lag1: float
    net_flow_lag2: float
    net_flow_lag24: float
    arrivals_lag1: float
    arrivals_lag2: float
    arrivals_lag3: float
    departures_lag1: float
    departures_lag2: float
    departures_lag3: float


class PredictRequest(BaseModel):
    """Stations to classify in one call."""

    instances: list[StationFeatures]


class Prediction(BaseModel):
    """Risk class of one station for the next hour."""

    station_id: int | None
    label: str
    label_id: int
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    """Predictions together with the model version and thresholds that produced them."""

    model_name: str | None
    model_version: str | None
    thresholds: dict[str, float]
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    """Registry version and lineage of the served model (registry_metadata.json)."""

    model_name: str | None
    model_version: str | None
    model_alias: str | None
    source_run_id: str | None
    git_commit: str | None
    data_version: str | None
    thresholds: dict[str, float]
    exported_at: str | None
