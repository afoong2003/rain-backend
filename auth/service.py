from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from argon2 import PasswordHasher


load_dotenv()

ph = PasswordHasher()

database_url = os.getenv("DATABASE_URL")

engine = create_engine(database_url)

def login_user(email: str, password: str) -> bool:
    try:
        hashed_password = ph.hash(password)

        with engine.connect() as connection:
            db_passsword = connection.execute(
                text("SELECT password FROM users WHERE email = :email"),
                {"email": email}
            ).fetchone()

        if (hashed_password == db_passsword):
            return True
        else: 
            return False
    except Exception as e:
        print(f"error: {e}")
    
     

def register_user(email: str, password: str) -> bool:
    hashed_password = ph.hash(password)
    






    