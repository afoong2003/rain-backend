import json
import logging
import os
from decimal import Decimal
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ml.plant_regression import add_engineered_features
from plant_data.plants import Plant, engine

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_price_model.pkl")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "plant_price_model_metadata.json")

# Global model cache
_model_cache = None
_metadata_cache = None


class PlantPredictionInput(BaseModel):
    """Input features for plant price prediction."""
    # Numeric features
    bloom_start: Optional[float] = None
    bloom_end: Optional[float] = None
    height_min: Optional[float] = None
    height_max: Optional[float] = None
    space_min: Optional[float] = None
    space_max: Optional[float] = None
    
    # Boolean features
    color_red: Optional[bool] = None
    color_orange: Optional[bool] = None
    color_yellow: Optional[bool] = None
    color_green: Optional[bool] = None
    color_blue: Optional[bool] = None
    color_purple: Optional[bool] = None
    color_pink: Optional[bool] = None
    color_brown: Optional[bool] = None
    color_white: Optional[bool] = None
    
    flood_tolerant: Optional[bool] = None
    drought_tolerant: Optional[bool] = None
    road_salt_tolerant: Optional[bool] = None
    
    sun_full: Optional[bool] = None
    sun_partial: Optional[bool] = None
    sun_shade: Optional[bool] = None
    
    moisture_wet: Optional[bool] = None
    moisture_med: Optional[bool] = None
    moisture_dry: Optional[bool] = None
    
    benefits_pollinators: Optional[bool] = None
    benefits_butterflies: Optional[bool] = None
    benefits_hummingbirds: Optional[bool] = None
    benefits_birds: Optional[bool] = None
    
    deer_resistant: Optional[bool] = None
    
    pos_base: Optional[bool] = None
    pos_slope: Optional[bool] = None
    pos_margin: Optional[bool] = None
    
    # Categorical features
    form: Optional[str] = None
    soil_pref: Optional[str] = None
    soil_ph: Optional[str] = None


class PredictionResponse(BaseModel):
    """Prediction response."""
    predicted_price_rating: float
    model_name: str
    confidence_r2: float


class RecommendationInput(BaseModel):
    """Input for plant recommendation."""
    budget: float = Field(default=10.0, gt=0)
    max_recommendations: int = Field(default=5, ge=1, le=20)
    min_popularity_rating: int = Field(default=0, ge=0)
    sun_full: bool = False
    sun_partial: bool = False
    sun_shade: bool = False
    moisture_wet: bool = False
    moisture_med: bool = False
    moisture_dry: bool = False
    pos_base: bool = False
    pos_slope: bool = False
    pos_margin: bool = False
    flood_tolerant: Optional[bool] = None
    drought_tolerant: Optional[bool] = None
    road_salt_tolerant: Optional[bool] = None
    soil_pref: Optional[str] = None
    soil_ph: Optional[str] = None
    space_min: Optional[float] = None
    space_max: Optional[float] = None
    benefits_pollinators: Optional[bool] = None
    benefits_butterflies: Optional[bool] = None
    benefits_hummingbirds: Optional[bool] = None
    benefits_birds: Optional[bool] = None


class RecommendedPlant(BaseModel):
    """Single plant recommendation."""
    plant_id: int
    display_name: str
    popularity_rating: int
    predicted_price: float
    score: float
    image: str
    score_breakdown: dict[str, float]
    matched_preferences: list[str]


class RecommendationResponse(BaseModel):
    """Recommendation response with ranked plants."""
    recommendations: list[RecommendedPlant]
    match_strategy: str = "closest_matches"


BOOLEAN_PREF_FIELDS = [
    "flood_tolerant",
    "drought_tolerant",
    "road_salt_tolerant",
    "benefits_pollinators",
    "benefits_butterflies",
    "benefits_hummingbirds",
    "benefits_birds",
]

GROUPED_PREFS = {
    "sun": ["sun_full", "sun_partial", "sun_shade"],
    "moisture": ["moisture_wet", "moisture_med", "moisture_dry"],
    "position": ["pos_base", "pos_slope", "pos_margin"],
}

RECOMMENDATION_WEIGHTS = {
    "price": 0.20,
    "popularity": 0.15,
    "tolerances": 0.21,
    "sun": 0.12,
    "moisture": 0.12,
    "position": 0.06,
    "space": 0.05,
    "soil": 0.04,
    "benefits": 0.05,
}


def normalize_numeric(value: Any, default: float = 0.0) -> float:
    """Convert database values like Decimal safely to floats."""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_plant_features(plant: Plant) -> dict[str, Any]:
    """Build the model feature dictionary for a plant row."""
    return {
        "bloom_start": normalize_numeric(plant.bloom_start),
        "bloom_end": normalize_numeric(plant.bloom_end),
        "height_min": normalize_numeric(plant.height_min),
        "height_max": normalize_numeric(plant.height_max),
        "space_min": normalize_numeric(plant.space_min),
        "space_max": normalize_numeric(plant.space_max),
        "color_red": bool(plant.color_red),
        "color_orange": bool(plant.color_orange),
        "color_yellow": bool(plant.color_yellow),
        "color_green": bool(plant.color_green),
        "color_blue": bool(plant.color_blue),
        "color_purple": bool(plant.color_purple),
        "color_pink": bool(plant.color_pink),
        "color_brown": bool(plant.color_brown),
        "color_white": bool(plant.color_white),
        "flood_tolerant": bool(plant.flood_tolerant),
        "drought_tolerant": bool(plant.drought_tolerant),
        "road_salt_tolerant": bool(plant.road_salt_tolerant),
        "sun_full": bool(plant.sun_full),
        "sun_partial": bool(plant.sun_partial),
        "sun_shade": bool(plant.sun_shade),
        "moisture_wet": bool(plant.moisture_wet),
        "moisture_med": bool(plant.moisture_med),
        "moisture_dry": bool(plant.moisture_dry),
        "benefits_pollinators": bool(plant.benefits_pollinators),
        "benefits_butterflies": bool(plant.benefits_butterflies),
        "benefits_hummingbirds": bool(plant.benefits_hummingbirds),
        "benefits_birds": bool(plant.benefits_birds),
        "deer_resistant": bool(plant.deer_resistant),
        "pos_base": bool(plant.pos_base),
        "pos_slope": bool(plant.pos_slope),
        "pos_margin": bool(plant.pos_margin),
        "form": plant.form or "unknown",
        "soil_pref": plant.soil_pref or "unknown",
        "soil_ph": plant.soil_ph or "unknown",
    }


def predict_price_for_plant(pipeline, plant: Plant) -> float:
    """Predict a price score for a plant using the trained model."""
    feature_df = pd.DataFrame([build_plant_features(plant)])
    prepared_df = prepare_features(feature_df)
    return float(pipeline.predict(prepared_df)[0])


def score_boolean_preference(
    requested: Optional[bool],
    actual: Optional[bool],
    label: str,
) -> tuple[Optional[float], Optional[str]]:
    """Score an optional boolean preference."""
    if requested is None:
        return None, None
    if requested is True:
        if bool(actual):
            return 1.0, label
        return 0.0, None
    if requested is False:
        return (1.0, None) if not bool(actual) else (0.25, None)
    return None, None


def score_group_match(
    request_data: RecommendationInput,
    plant: Plant,
    fields: list[str],
    labels: dict[str, str],
) -> tuple[Optional[float], list[str]]:
    """Score grouped preferences like sun or moisture."""
    selected_fields = [field for field in fields if getattr(request_data, field)]
    if not selected_fields:
        return None, []

    matched = [field for field in selected_fields if bool(getattr(plant, field))]
    if matched:
        return len(matched) / len(selected_fields), [labels[field] for field in matched]
    return 0.0, []


def score_space_preference(
    request_data: RecommendationInput,
    plant: Plant,
) -> tuple[Optional[float], Optional[str]]:
    """Score how well the requested space range overlaps the plant spacing."""
    desired_min = request_data.space_min
    desired_max = request_data.space_max
    if desired_min is None and desired_max is None:
        return None, None

    plant_min = normalize_numeric(plant.space_min, default=-1.0)
    plant_max = normalize_numeric(plant.space_max, default=-1.0)
    if plant_min < 0 and plant_max < 0:
        return 0.25, None
    if plant_min < 0:
        plant_min = plant_max
    if plant_max < 0:
        plant_max = plant_min

    desired_min = plant_min if desired_min is None else desired_min
    desired_max = plant_max if desired_max is None else desired_max
    overlap = max(0.0, min(plant_max, desired_max) - max(plant_min, desired_min))
    desired_width = max(desired_max - desired_min, 1.0)
    if overlap > 0:
        return min(1.0, overlap / desired_width), "space fit"

    plant_center = (plant_min + plant_max) / 2
    desired_center = (desired_min + desired_max) / 2
    distance = abs(plant_center - desired_center)
    return max(0.0, 1.0 - (distance / max(desired_width, 1.0))), None


def score_text_match(requested: Optional[str], actual: Optional[str]) -> Optional[float]:
    """Score a text preference using normalized containment."""
    if not requested:
        return None
    if not actual:
        return 0.0

    requested_tokens = {token for token in requested.lower().replace(",", " ").split() if token}
    actual_tokens = {token for token in actual.lower().replace(",", " ").split() if token}
    if not requested_tokens:
        return None
    if requested.lower() in actual.lower():
        return 1.0
    return len(requested_tokens & actual_tokens) / len(requested_tokens)


def build_recommendation_scores(
    request_data: RecommendationInput,
    plant: Plant,
    predicted_price: float,
) -> tuple[float, dict[str, float], list[str]]:
    """Combine all preference signals into one explainable recommendation score."""
    score_breakdown: dict[str, float] = {}
    matched_preferences: list[str] = []

    effective_budget = max(request_data.budget, 0.1)
    price_score = min(effective_budget / max(predicted_price, 0.1), 1.0)
    if predicted_price <= effective_budget:
        matched_preferences.append("within budget")
    score_breakdown["price"] = price_score

    popularity = normalize_numeric(plant.popularity_rating)
    popularity_target = max(request_data.min_popularity_rating, 1)
    popularity_score = min(popularity / max(popularity_target, 1), 1.0)
    if popularity >= request_data.min_popularity_rating and popularity > 0:
        matched_preferences.append("popular choice")
    score_breakdown["popularity"] = popularity_score

    tolerance_scores = []
    for field in ["flood_tolerant", "drought_tolerant", "road_salt_tolerant"]:
        field_score, label = score_boolean_preference(
            getattr(request_data, field),
            getattr(plant, field),
            Plant.tag_map.get(field, field),
        )
        if field_score is not None:
            tolerance_scores.append(field_score)
        if label:
            matched_preferences.append(label)
    score_breakdown["tolerances"] = (
        sum(tolerance_scores) / len(tolerance_scores) if tolerance_scores else 0.5
    )

    sun_score, sun_matches = score_group_match(
        request_data,
        plant,
        GROUPED_PREFS["sun"],
        Plant.tag_map,
    )
    score_breakdown["sun"] = 0.5 if sun_score is None else sun_score
    matched_preferences.extend(sun_matches)

    moisture_score, moisture_matches = score_group_match(
        request_data,
        plant,
        GROUPED_PREFS["moisture"],
        Plant.tag_map,
    )
    score_breakdown["moisture"] = 0.5 if moisture_score is None else moisture_score
    matched_preferences.extend(moisture_matches)

    position_score, position_matches = score_group_match(
        request_data,
        plant,
        GROUPED_PREFS["position"],
        Plant.tag_map,
    )
    score_breakdown["position"] = 0.5 if position_score is None else position_score
    matched_preferences.extend(position_matches)

    space_score, space_match = score_space_preference(request_data, plant)
    score_breakdown["space"] = 0.5 if space_score is None else space_score
    if space_match:
        matched_preferences.append(space_match)

    soil_pref_score = score_text_match(request_data.soil_pref, plant.soil_pref)
    soil_ph_score = score_text_match(request_data.soil_ph, plant.soil_ph)
    soil_components = [score for score in [soil_pref_score, soil_ph_score] if score is not None]
    score_breakdown["soil"] = (
        sum(soil_components) / len(soil_components) if soil_components else 0.5
    )
    if soil_pref_score and soil_pref_score > 0:
        matched_preferences.append("soil preference match")
    if soil_ph_score and soil_ph_score > 0:
        matched_preferences.append("soil pH match")

    benefit_scores = []
    for field in [
        "benefits_pollinators",
        "benefits_butterflies",
        "benefits_hummingbirds",
        "benefits_birds",
    ]:
        field_score, label = score_boolean_preference(
            getattr(request_data, field),
            getattr(plant, field),
            Plant.tag_map.get(field, field),
        )
        if field_score is not None:
            benefit_scores.append(field_score)
        if label:
            matched_preferences.append(label)
    score_breakdown["benefits"] = (
        sum(benefit_scores) / len(benefit_scores) if benefit_scores else 0.5
    )

    weighted_score = sum(
        score_breakdown[name] * weight for name, weight in RECOMMENDATION_WEIGHTS.items()
    )
    return weighted_score, score_breakdown, sorted(set(matched_preferences))


def load_model():
    """Load trained model and metadata from disk."""
    global _model_cache, _metadata_cache
    
    if _model_cache is not None and _metadata_cache is not None:
        return _model_cache, _metadata_cache
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the model first using plant_regression.py"
        )
    
    try:
        _model_cache = joblib.load(MODEL_PATH)
        with open(MODEL_METADATA_PATH, "r") as f:
            _metadata_cache = json.load(f)
        logger.info("Model loaded successfully")
        return _model_cache, _metadata_cache
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def prepare_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    return add_engineered_features(input_df.copy())


router = APIRouter(
    prefix="/predict",
    tags=["Predictions"]
)


@router.get("/health")
async def model_health():
    """Check if the model is available."""
    try:
        load_model()
        return {"status": "healthy", "message": "Model is ready"}
    except HTTPException as e:
        return {"status": "unavailable", "message": e.detail}


@router.post("/price", response_model=PredictionResponse)
async def predict_plant_price(input_data: PlantPredictionInput) -> PredictionResponse:
    """
    Predict plant price rating based on plant characteristics.
    
    Expects plant features and returns predicted price rating (0-5 scale typically).
    """
    try:
        pipeline, metadata = load_model()
        
        # Convert input to DataFrame in the same format as training
        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict])
        input_df = prepare_features(input_df)
        
        # Make prediction
        prediction = pipeline.predict(input_df)[0]
        
        return PredictionResponse(
            predicted_price_rating=float(prediction),
            model_name=metadata.get("model_name", "Unknown"),
            confidence_r2=float(metadata.get("r2_score", 0.0))
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


recommendations_router = APIRouter(
    prefix="/recommend",
    tags=["Recommendations"]
)


@recommendations_router.post("/plants", response_model=RecommendationResponse)
async def recommend_plants(input_data: RecommendationInput) -> RecommendationResponse:
    """
    Get ranked plant recommendations based on garden preferences.

    The score blends budget fit, popularity, environmental tolerances,
    sun, moisture, position, spacing, soil, and habitat benefits.
    """
    recommendations = []
    try:
        plants = await Plant.get_records(engine)
        if input_data.min_popularity_rating > 0:
            plants = [
                plant
                for plant in plants
                if normalize_numeric(plant.popularity_rating) >= input_data.min_popularity_rating
            ]

        if not plants:
            return RecommendationResponse(recommendations=[], match_strategy="closest_matches")

        pipeline, _ = load_model()

        for plant in plants:
            predicted_price = predict_price_for_plant(pipeline, plant)
            score, score_breakdown, matched_preferences = build_recommendation_scores(
                input_data, plant, predicted_price
            )

            recommendations.append(
                RecommendedPlant(
                    plant_id=plant.plant_id,
                    display_name=plant.display_name or "Unknown",
                    popularity_rating=plant.popularity_rating or 0,
                    predicted_price=predicted_price,
                    score=score,
                    image=plant.image or "",
                    score_breakdown=score_breakdown,
                    matched_preferences=matched_preferences,
                )
            )

        max_recommendations = max(1, min(input_data.max_recommendations, 20))
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return RecommendationResponse(
            recommendations=recommendations[:max_recommendations],
            match_strategy="closest_matches",
        )
    
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
