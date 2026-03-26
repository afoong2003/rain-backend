import json
import logging
import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
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
    sun_full: bool = False
    sun_partial: bool = False
    sun_shade: bool = False
    moisture_wet: bool = False
    moisture_med: bool = False
    moisture_dry: bool = False
    pos_base: bool = False
    pos_slope: bool = False
    pos_margin: bool = False
    budget: float = 10.0


class RecommendedPlant(BaseModel):
    """Single plant recommendation."""
    plant_id: int
    display_name: str
    popularity_rating: int
    predicted_price: float
    score: float
    image: str


class RecommendationResponse(BaseModel):
    """Recommendation response with top 2 plants."""
    recommendations: list[RecommendedPlant]


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
    Get top 2 plant recommendations based on user preferences.
    
    Scoring = popularity_rating / predicted_price
    Budget is soft preference (expensive plants get penalized).
    """
    # Build filter conditions
    filters = []
    if input_data.sun_full:
        filters.append(Plant.sun_full == True)
    if input_data.sun_partial:
        filters.append(Plant.sun_partial == True)
    if input_data.sun_shade:
        filters.append(Plant.sun_shade == True)
    if input_data.moisture_wet:
        filters.append(Plant.moisture_wet == True)
    if input_data.moisture_med:
        filters.append(Plant.moisture_med == True)
    if input_data.moisture_dry:
        filters.append(Plant.moisture_dry == True)
    if input_data.pos_base:
        filters.append(Plant.pos_base == True)
    if input_data.pos_slope:
        filters.append(Plant.pos_slope == True)
    if input_data.pos_margin:
        filters.append(Plant.pos_margin == True)
    
    recommendations = []
    try:
        async with AsyncSession(engine) as session:
            query = select(Plant).where(or_(*filters)) if filters else select(Plant)
            results = await session.execute(query)
            plants = results.scalars().all()
            
            if not plants:
                return RecommendationResponse(recommendations=[])
            
            # Load model for price predictions
            pipeline, metadata = load_model()
            
            for plant in plants:
                # Build feature dict from plant data
                features = {
                    "bloom_start": float(plant.bloom_start) if plant.bloom_start else 0,
                    "bloom_end": float(plant.bloom_end) if plant.bloom_end else 0,
                    "height_min": float(plant.height_min) if plant.height_min else 0,
                    "height_max": float(plant.height_max) if plant.height_max else 0,
                    "space_min": float(plant.space_min) if plant.space_min else 0,
                    "space_max": float(plant.space_max) if plant.space_max else 0,
                    "color_red": plant.color_red or False,
                    "color_orange": plant.color_orange or False,
                    "color_yellow": plant.color_yellow or False,
                    "color_green": plant.color_green or False,
                    "color_blue": plant.color_blue or False,
                    "color_purple": plant.color_purple or False,
                    "color_pink": plant.color_pink or False,
                    "color_brown": plant.color_brown or False,
                    "color_white": plant.color_white or False,
                    "flood_tolerant": plant.flood_tolerant or False,
                    "drought_tolerant": plant.drought_tolerant or False,
                    "road_salt_tolerant": plant.road_salt_tolerant or False,
                    "sun_full": plant.sun_full or False,
                    "sun_partial": plant.sun_partial or False,
                    "sun_shade": plant.sun_shade or False,
                    "moisture_wet": plant.moisture_wet or False,
                    "moisture_med": plant.moisture_med or False,
                    "moisture_dry": plant.moisture_dry or False,
                    "benefits_pollinators": plant.benefits_pollinators or False,
                    "benefits_butterflies": plant.benefits_butterflies or False,
                    "benefits_hummingbirds": plant.benefits_hummingbirds or False,
                    "benefits_birds": plant.benefits_birds or False,
                    "deer_resistant": plant.deer_resistant or False,
                    "pos_base": plant.pos_base or False,
                    "pos_slope": plant.pos_slope or False,
                    "pos_margin": plant.pos_margin or False,
                    "form": plant.form or "unknown",
                    "soil_pref": plant.soil_pref or "unknown",
                    "soil_ph": plant.soil_ph or "neutral",
                }
                
                feature_df = pd.DataFrame([features])
                predicted_price = float(pipeline.predict(feature_df)[0])
                
                # Score = rating / price with budget soft penalty
                rating = float(plant.popularity_rating) if plant.popularity_rating else 1.0
                base_score = rating / max(predicted_price, 0.1)
                
                # Soft penalty if over budget
                if predicted_price > input_data.budget:
                    budget_penalty = predicted_price / input_data.budget
                    score = base_score / budget_penalty
                else:
                    score = base_score
                
                recommendations.append(
                    RecommendedPlant(
                        plant_id=plant.plant_id,
                        display_name=plant.display_name or "Unknown",
                        popularity_rating=plant.popularity_rating or 0,
                        predicted_price=predicted_price,
                        score=score,
                        image=plant.image or ""
                    )
                )
            
            # Sort by score (highest first) and return top 2
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return RecommendationResponse(recommendations=recommendations[:2])
    
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
