from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from argon2 import PasswordHasher

load_dotenv()

ph = PasswordHasher()

database_url = os.getenv("DATABASE_URL")

engine = create_engine(database_url)

def login_user(email: str, password: str) -> bool:
    """Verify user credentials. Returns True if email and password match."""
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT password FROM users WHERE email = :email"),
                {"email": email}
            ).fetchone()

        if not result:
            return False
        
        db_password_hash = result[0]
        
        # Verify the provided password against the stored hash
        try:
            ph.verify(db_password_hash, password)
            return True
        except Exception:
            return False
    except Exception as e:
        print(f"error: {e}")
        return False
    
     

def register_user(email: str, password: str) -> bool:
    """Register a new user with hashed password."""
    try:
        hashed_password = ph.hash(password)
        
        with engine.connect() as connection:
            connection.execute(
                text("INSERT INTO users (email, password) VALUES (:email, :password)"),
                {"email": email, "password": hashed_password}
            )
            connection.commit()
        
        return True
    except Exception as e:
        print(f"Registration error: {e}")
        return False
    






    