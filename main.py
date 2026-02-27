from fastapi import FastAPI, APIRouter
from plant_data import plants as plant_query
from auth import service

app = FastAPI()

plants_router = APIRouter(
    prefix="/plants",
    tags=["Plants"] 
)

@plants_router.get("/preview")
async def preview_plants() -> list:
    preview = await plant_query.Plant.get_all_plants(plant_query.engine)
    return preview

@plants_router.get("/get_by_plant_id")
def get_specific_plant(id: int) -> list:
    pass

auth_router = APIRouter(
    prefix="/auth",
    tags=["Auth"]
)

@auth_router.post("/login")
async def user_login(username: str, password: str) -> bool:
    pass

@auth_router.post("/register")
async def register_user(username: str, password: str) -> bool:
    pass

app.include_router(plants_router)
app.include_router(auth_router)