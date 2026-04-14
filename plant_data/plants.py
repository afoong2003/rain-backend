from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PLANTS_CSV = BASE_DIR / "plant_data" / "plants.csv"
engine = None


def _normalize_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _normalize_number(value: Any) -> float | int | None:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _normalize_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


@dataclass
class PlantRecord:
    plant_id: int
    usda_id: str | None = None
    scientific_name: str | None = None
    display_name: str | None = None
    form: str | None = None
    price_rating: str | None = None
    colors: str | None = None
    bloom_start: float | int | None = None
    bloom_end: float | int | None = None
    soil_pref: str | None = None
    soil_ph: str | None = None
    warnings: str | None = None
    description: str | None = None
    image: str | None = None
    popularity_rating: int | None = None
    height_min: float | int | None = None
    height_max: float | int | None = None
    space_min: float | int | None = None
    space_max: float | int | None = None
    drought_tolerant: bool | None = None
    flood_tolerant: bool | None = None
    road_salt_tolerant: bool | None = None
    sun_full: bool | None = None
    sun_partial: bool | None = None
    sun_shade: bool | None = None
    moisture_wet: bool | None = None
    moisture_med: bool | None = None
    moisture_dry: bool | None = None
    benefits_pollinators: bool | None = None
    benefits_butterflies: bool | None = None
    benefits_hummingbirds: bool | None = None
    benefits_birds: bool | None = None
    deer_resistant: bool | None = None
    pos_base: bool | None = None
    pos_slope: bool | None = None
    pos_margin: bool | None = None
    color_red: bool | None = None
    color_orange: bool | None = None
    color_yellow: bool | None = None
    color_green: bool | None = None
    color_blue: bool | None = None
    color_purple: bool | None = None
    color_pink: bool | None = None
    color_brown: bool | None = None
    color_white: bool | None = None


class Plant:
    tag_map: ClassVar[dict[str, str]] = {
        "drought_tolerant": "Drought Tolerant",
        "flood_tolerant": "Flood Tolerant",
        "road_salt_tolerant": "Road Salt Tolerant",
        "sun_full": "Sun Full",
        "sun_partial": "Sun Partial",
        "sun_shade": "Sun Shade",
        "moisture_wet": "Moisture Wet",
        "moisture_med": "Moisture Medium",
        "moisture_dry": "Moisture Dry",
        "benefits_pollinators": "Benefit Pollinators",
        "benefits_butterflies": "Benefit Butterflies",
        "benefits_hummingbirds": "Benefit Hummingbirds",
        "benefits_birds": "Benefit Birds",
        "deer_resistant": "Deer Resistant",
        "color_red": "Red",
        "color_orange": "Orange",
        "color_yellow": "Yellow",
        "color_green": "Green",
        "color_blue": "Blue",
        "color_purple": "Purple",
        "color_pink": "Pink",
        "color_brown": "Brown",
        "color_white": "White",
        "pos_base": "Base",
        "pos_slope": "Slope",
        "pos_margin": "Margin",
    }

    _cache: ClassVar[list[PlantRecord] | None] = None
    _cache_mtime: ClassVar[float | None] = None

    @classmethod
    def csv_path(cls) -> Path:
        return DEFAULT_PLANTS_CSV

    @classmethod
    def _plant_tags(cls, plant: PlantRecord) -> list[str]:
        return [
            tag_name
            for field_name, tag_name in cls.tag_map.items()
            if getattr(plant, field_name, None) is True
        ]

    @classmethod
    def _build_plant_record(cls, row: dict[str, Any], fallback_id: int) -> PlantRecord:
        normalized: dict[str, Any] = {}
        text_fields = {
            "usda_id",
            "scientific_name",
            "display_name",
            "form",
            "price_rating",
            "colors",
            "soil_pref",
            "soil_ph",
            "warnings",
            "description",
            "image",
        }
        number_fields = {
            "plant_id",
            "bloom_start",
            "bloom_end",
            "popularity_rating",
            "height_min",
            "height_max",
            "space_min",
            "space_max",
        }
        bool_fields = set(cls.tag_map.keys())

        for field_name in PlantRecord.__dataclass_fields__:
            raw_value = row.get(field_name)
            if field_name in text_fields:
                normalized[field_name] = _normalize_text(raw_value)
            elif field_name in number_fields:
                normalized[field_name] = _normalize_number(raw_value)
            elif field_name in bool_fields:
                normalized[field_name] = _normalize_bool(raw_value)
            else:
                normalized[field_name] = raw_value

        plant_id = normalized.get("plant_id")
        normalized["plant_id"] = int(plant_id) if plant_id is not None else fallback_id
        popularity = normalized.get("popularity_rating")
        normalized["popularity_rating"] = int(popularity) if popularity is not None else None
        return PlantRecord(**normalized)

    @classmethod
    def load_records(cls) -> list[PlantRecord]:
        csv_path = cls.csv_path()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Plant CSV not found. Add your dataset at {csv_path}"
            )

        modified_time = csv_path.stat().st_mtime
        if cls._cache is not None and cls._cache_mtime == modified_time:
            return cls._cache

        df = pd.read_csv(csv_path)
        records = [
            cls._build_plant_record(row, fallback_id=index + 1)
            for index, row in enumerate(df.to_dict(orient="records"))
        ]
        cls._cache = records
        cls._cache_mtime = modified_time
        return records

    @classmethod
    async def get_records(cls, _engine=None) -> list[PlantRecord]:
        return cls.load_records()

    @classmethod
    async def get_all_plants(cls, _engine=None) -> list:
        plant_list = []
        for plant in cls.load_records():
            plant_list.append(
                {
                    "plant_id": plant.plant_id,
                    "display_name": plant.display_name,
                    "scientific_name": plant.scientific_name,
                    "popularity_rating": plant.popularity_rating,
                    "price_rating": plant.price_rating,
                    "image": plant.image,
                    "tags": cls._plant_tags(plant),
                }
            )
        return plant_list

    @classmethod
    async def get_plant_by_id(cls, _engine, target_id: int) -> list:
        for plant in cls.load_records():
            if plant.plant_id == target_id:
                return [
                    {
                        "display_name": plant.display_name,
                        "scientific_name": plant.scientific_name,
                        "price_rating": plant.price_rating,
                        "description": plant.description,
                        "bloom_start": plant.bloom_start,
                        "bloom_end": plant.bloom_end,
                        "height_min": plant.height_min,
                        "height_max": plant.height_max,
                        "image": plant.image,
                        "tags": cls._plant_tags(plant),
                    }
                ]
        return []

    @classmethod
    async def search_plant(cls, _engine, query: str) -> list:
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        matches = []
        for plant in cls.load_records():
            scientific_name = (plant.scientific_name or "").lower()
            display_name = (plant.display_name or "").lower()
            if scientific_name.startswith(normalized_query) or display_name.startswith(normalized_query):
                matches.append(
                    {
                        "plant_id": plant.plant_id,
                        "display_name": plant.display_name,
                        "scientific_name": plant.scientific_name,
                        "image": plant.image,
                    }
                )
        return matches

    @classmethod
    async def plant_filter(
        cls,
        _engine,
        sun_shade: bool | None = None,
        sun_full: bool | None = None,
        sun_partial: bool | None = None,
        moisture_wet: bool | None = None,
        moisture_dry: bool | None = None,
        moisture_med: bool | None = None,
        pos_base: bool | None = None,
        pos_slope: bool | None = None,
        pos_margin: bool | None = None,
    ) -> list:
        active_filters = {
            "sun_full": sun_full,
            "sun_shade": sun_shade,
            "sun_partial": sun_partial,
            "moisture_wet": moisture_wet,
            "moisture_dry": moisture_dry,
            "moisture_med": moisture_med,
            "pos_base": pos_base,
            "pos_slope": pos_slope,
            "pos_margin": pos_margin,
        }

        filtered = []
        for plant in cls.load_records():
            if any(
                value is True and getattr(plant, field_name) is not True
                for field_name, value in active_filters.items()
            ):
                continue
            filtered.append(
                {
                    "plant_id": plant.plant_id,
                    "display_name": plant.display_name,
                    "scientific_name": plant.scientific_name,
                    "image": plant.image,
                }
            )
        return filtered
