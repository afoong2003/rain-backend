import os
from typing import ClassVar
from dotenv import load_dotenv
from sqlalchemy import Text, Integer, select, String, Numeric, Boolean, or_
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

load_dotenv()
database_url = os.getenv("DATABASE_URL")
engine = create_async_engine(database_url, echo=False)

class Base(DeclarativeBase):
    pass

class Plant(Base):
    __tablename__ = "plants"
    plant_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    usda_id: Mapped[str] = mapped_column(String(50), nullable=True)
    scientific_name: Mapped[str] = mapped_column(String(150), nullable=True)
    display_name: Mapped[str] = mapped_column(String(150), nullable=True)
    form: Mapped[str] = mapped_column(String(50), nullable=True)
    price_rating: Mapped[str] = mapped_column(String(50), nullable=True)
    colors: Mapped[str] = mapped_column(String(150), nullable=True)
    bloom_start: Mapped[str] = mapped_column(String(50), nullable=True)
    bloom_end: Mapped[str] = mapped_column(String(50), nullable=True)
    soil_pref: Mapped[str] = mapped_column(String(150), nullable=True)
    soil_ph: Mapped[str] = mapped_column(String(100), nullable=True)
    warnings: Mapped[str] = mapped_column(Text, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    image: Mapped[str] = mapped_column(Text, nullable=True)

    popularity_rating: Mapped[int] = mapped_column(Integer, nullable=True)
    height_min: Mapped[float] = mapped_column(Numeric, nullable=True)
    height_max: Mapped[float] = mapped_column(Numeric, nullable=True)
    space_min: Mapped[float] = mapped_column(Numeric, nullable=True)
    space_max: Mapped[float] = mapped_column(Numeric, nullable=True)

    drought_tolerant: Mapped[bool] = mapped_column(Boolean, nullable=True)
    flood_tolerant: Mapped[bool] = mapped_column(Boolean, nullable=True)
    road_salt_tolerant: Mapped[bool] = mapped_column(Boolean, nullable=True)
    sun_full: Mapped[bool] = mapped_column(Boolean, nullable=True)
    sun_partial: Mapped[bool] = mapped_column(Boolean, nullable=True)
    sun_shade: Mapped[bool] = mapped_column(Boolean, nullable=True)
    moisture_wet: Mapped[bool] = mapped_column(Boolean, nullable=True)
    moisture_med: Mapped[bool] = mapped_column(Boolean, nullable=True)
    moisture_dry: Mapped[bool] = mapped_column(Boolean, nullable=True)
    benefits_pollinators: Mapped[bool] = mapped_column(Boolean, nullable=True)
    benefits_butterflies: Mapped[bool] = mapped_column(Boolean, nullable=True)
    benefits_hummingbirds: Mapped[bool] = mapped_column(Boolean, nullable=True)
    benefits_birds: Mapped[bool] = mapped_column(Boolean, nullable=True)
    deer_resistant: Mapped[bool] = mapped_column(Boolean, nullable=True)
    
    color_red: Mapped[bool] = mapped_column(Boolean, nullable=True)
    color_orange: Mapped[bool] = mapped_column(Boolean, nullable=True)
    color_yellow: Mapped[bool] = mapped_column(Boolean, nullable=True)
    color_green: Mapped[bool] = mapped_column(Boolean, nullable=True)
    color_blue: Mapped[bool] = mapped_column(Boolean, nullable=True)
    color_purple: Mapped[bool] = mapped_column(Boolean, nullable=True)
    color_pink: Mapped[bool] = mapped_column(Boolean, nullable=True)
    color_brown: Mapped[bool] = mapped_column(Boolean, nullable=True)
    color_white: Mapped[bool] = mapped_column(Boolean, nullable=True)

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
            "color_white": "White"
        }

    @classmethod
    async def get_all_plants(cls, engine) -> list:
        try:
            async with AsyncSession(engine) as session:
                query = select(cls.plant_id, cls.display_name, cls.scientific_name, 
                               cls.popularity_rating, cls.form, cls.price_rating, 
                               cls.description, cls.image, cls.sun_full, 
                               cls.benefits_birds, cls.benefits_butterflies, cls.benefits_pollinators, 
                               cls.benefits_hummingbirds, cls.drought_tolerant, cls.flood_tolerant, 
                               cls.road_salt_tolerant, cls.sun_partial, cls.sun_shade,
                               cls.moisture_dry, cls.moisture_med, cls.moisture_wet, 
                               cls.deer_resistant
                               )
                
                results = await session.execute(query)
                plant_list = []
                for row in results:
                    plant_tags = []

                    if (row.form):
                        plant_tags.append(row.form)
                    
                    for col_name, tag_name in cls.tag_map.items():
                        if row._mapping.get(col_name) is True:
                            plant_tags.append(tag_name)

                    plant_list.append(
                        {
                        "plant_id": row.plant_id,
                        "display_name": row.display_name,
                        "scientific_name": row.scientific_name,
                        "popularity_rating": row.popularity_rating,
                        "price_rating": row.price_rating,
                        "description": row.description,
                        "image": row.image,
                        "tags": plant_tags
                        }
                    )
                    
                return plant_list
            
        except Exception as e:
            print(f"err: {e}")
            return []
            
    @classmethod
    async def get_plant_by_id(cls, engine, target_id: int) -> list:
        try:
            async with AsyncSession(engine) as session:
                search_query = (
                    select(
                        cls.display_name, cls.scientific_name, cls.price_rating,
                        cls.description, cls.moisture_dry, cls.moisture_med,
                        cls.moisture_wet, cls.bloom_end, cls.bloom_start, 
                        cls.height_min, cls.height_max, cls.sun_full, cls.sun_partial,
                        cls.sun_shade, cls.image
                        )
                        .where(
                            cls.plant_id == target_id
                        )
                )
                result = await session.execute(search_query)
                plant = []

                for row in result:
                    plant_tags = []
                 
                    for col_name, tag_name in cls.tag_map.items():
                        if row._mapping.get(col_name) is True:
                            plant_tags.append(tag_name)
                        
                    plant.append(
                        {
                            "display_name": row.display_name,
                            "scientific_name": row.scientific_name,
                            "price_rating": row.price_rating,
                            "description": row.description,
                            "bloom_start": row.bloom_start,
                            "bloom_end": row.bloom_end,
                            "height_min": row.height_min,
                            "height_max": row.height_max,
                            "image": row.image,
                            "tags": plant_tags
                        }
                    )
                return plant

        except Exception as e:
            print(e)
            return []

    @classmethod
    async def search_plant(cls, engine, query: str) -> list:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        try:
            async with AsyncSession(engine) as session:
                starts_with = f"{normalized_query}%"
                search_query = (
                    select(
                        cls.plant_id, cls.display_name, cls.scientific_name,
                        cls.popularity_rating, cls.form, cls.price_rating,
                        cls.description, cls.image, cls.sun_full,
                        cls.benefits_birds, cls.benefits_butterflies,
                        cls.benefits_pollinators, cls.benefits_hummingbirds,
                        cls.drought_tolerant, cls.flood_tolerant,
                        cls.road_salt_tolerant, cls.sun_partial,
                        cls.sun_shade, cls.moisture_dry, cls.moisture_med,
                        cls.moisture_wet, cls.deer_resistant
                    )
                    .where(
                        or_(
                            cls.scientific_name.ilike(starts_with),
                            cls.display_name.ilike(starts_with),
                        )
                    )
                )


                results = await session.execute(search_query)
                plant_list = []

                for row in results:
                    plant_tags = []

                    if row.form:
                        plant_tags.append(row.form)

                    for col_name, tag_name in cls.tag_map.items():
                        if row._mapping.get(col_name) is True:
                            plant_tags.append(tag_name)

                    plant_list.append(
                        {
                            "plant_id": row.plant_id,
                            "display_name": row.display_name,
                            "scientific_name": row.scientific_name,
                            "popularity_rating": row.popularity_rating,
                            "price_rating": row.price_rating,
                            "description": row.description,
                            "image": row.image,
                            "tags": plant_tags,
                        }
                    )


                return plant_list

        except Exception as e:
            print(f"err: {e}")
            return []

