from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from argon2 import PasswordHasher


load_dotenv()

ph = PasswordHasher()

database_url = os.getenv("DATABASE_URL")

engine = create_engine(database_url)

def login_user(username: str, password: str) -> bool:
    pass 

def register_user(username: str, password: str) -> bool:
    pass






    