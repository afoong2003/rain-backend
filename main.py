from fastapi import FastAPI, APIRouter
from plant_data import plants as plant_query
from auth import service
from pydantic import BaseModel

app = FastAPI()

class UserCredentials(BaseModel):
    username: str
    password: str

class PlantFilter(BaseModel):
    sun_full: bool | None = None
    sun_partial: bool | None = None
    sun_shade: bool | None = None
    moisture_wet: bool | None = None
    moisture_medium: bool | None = None
    moisture_dry: bool | None = None
    pos_base: bool | None = None
    pos_slope: bool | None = None
    pos_margin: bool | None = None
    

plants_router = APIRouter(
    prefix="/plants",
    tags=["Plants"] 
)

@plants_router.get("/preview")
async def preview_plants() -> list:
    preview = await plant_query.Plant.get_all_plants(plant_query.engine)
    return preview

@plants_router.get("/search")
async def search_plants(q: str) -> list:
    return await plant_query.Plant.search_plant(plant_query.engine, q)

@plants_router.post("/filter")
async def get_filtered_plant(filters: PlantFilter) -> list:
    return await plant_query.Plant.plant_filter(
        plant_query.engine, 
        sun_full=filters.sun_full,
        sun_partial=filters.sun_partial,
        sun_shade=filters.sun_shade,
        moisture_wet=filters.moisture_wet,
        moisture_dry=filters.moisture_dry,
        moisture_med=filters.moisture_medium,
        pos_base=filters.pos_base,
        pos_slope=filters.pos_slope,
        pos_margin=filters.pos_margin
        )

@plants_router.get("/{plant_id}")
async def get_specific_plant(plant_id: int) -> list:
    return await plant_query.Plant.get_plant_by_id(plant_query.engine, plant_id)


auth_router = APIRouter(
    prefix="/auth",
    tags=["Auth"]
)

@auth_router.post("/login")
async def user_login(user: UserCredentials) -> bool:
    pass

@auth_router.post("/register")
async def register_user(user: UserCredentials) -> bool:
    pass

app.include_router(plants_router)
app.include_router(auth_router)