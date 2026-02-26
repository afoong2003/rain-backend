import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Text, Integer, select, String, Numeric, Boolean
from sqlalchemy.orm import Session, Mapped, mapped_column, DeclarativeBase

load_dotenv()
database_url = os.getenv("DATABASE_URL")
engine = create_engine(database_url)

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

    popularity: Mapped[int] = mapped_column(Integer, nullable=True)
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

    def to_dict(self):
        plant_data = {}

        for column in self.__table__.columns:
            column_name = column.name
            value = getattr(self, column_name)
            plant_data[column_name] = value
        return plant_data
        

    @classmethod
    def get_all_plants(cls, engine):
        try:
            with Session(engine) as session:
                query = select(cls.plant_id, cls.display_name, cls.scientific_name, cls.popularity, cls.form, cls.price_rating).limit(10)
                results = session.execute(query)
                plant_list = []
                for row in results:
                    plant_list.append({
                        "plant_id": row.plant_id,
                        "display_name": row.display_name,
                        "scientific_name": row.scientific_name,
                        "popularity": row.popularity,
                        "form": row.form,
                        "price_rating": row.price_rating
                    })
                return plant_list
        except Exception as e:
            print(f"err: {e}")
            
    
    @classmethod
    def get_plant_by_id(cls, engine, target_id: int):
        pass

test = Plant.get_all_plants(engine)
print(test)
