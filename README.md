# rain-backend

FastAPI backend for the Rain Garden App using a local CSV plant dataset and ML-based price prediction.

## Setup

### 1. Add your dataset

Put your plant dataset at `plant_data/plants.csv`.

The backend expects columns used by the APIs and model, such as:

```text
plant_id,display_name,scientific_name,popularity_rating,price_rating,image,
flood_tolerant,drought_tolerant,road_salt_tolerant,
sun_full,sun_partial,sun_shade,
moisture_wet,moisture_med,moisture_dry,
pos_base,pos_slope,pos_margin,
space_min,space_max,soil_pref,soil_ph,
benefits_pollinators,benefits_butterflies,benefits_hummingbirds,benefits_birds
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the ML model

```bash
python ml/plant_regression.py --source csv
```

You can also point to a different file:

```bash
python ml/plant_regression.py --source csv --csv-path path/to/plants.csv
```

This creates:
- `ml/models/plant_price_model.pkl`
- `ml/models/plant_price_model_metadata.json`

### 4. Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API endpoints

### Plants

`GET /plants/preview`

`GET /plants/search?q=rose`

`GET /plants/{plant_id}`

`POST /plants/filter`

Example:

```json
{
  "sun_full": true,
  "moisture_wet": true,
  "pos_base": true
}
```

### Predictions

`GET /predict/health`

`POST /predict/price`

### Recommendations

`POST /recommend/plants`

Example:

```json
{
  "budget": 8,
  "max_recommendations": 5,
  "min_popularity_rating": 3,
  "flood_tolerant": true,
  "road_salt_tolerant": true,
  "sun_full": true,
  "moisture_wet": true,
  "pos_base": true,
  "space_min": 18,
  "space_max": 36,
  "soil_pref": "loam",
  "soil_ph": "neutral",
  "benefits_pollinators": true,
  "benefits_birds": true
}
```

## Notes

- `.env`, PostgreSQL, and pgAdmin are no longer required for the plant API flow.
- Auth routes are not mounted in the app right now.
- If `plant_data/plants.csv` is missing, the plant endpoints will return an error until you add it.
