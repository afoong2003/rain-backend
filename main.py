from fastapi import FastAPI
from auth.service import authenticate_user, register_user
from plants.plants import engine as plants_engine

app = FastAPI()


