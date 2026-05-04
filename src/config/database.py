from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from .settings import Settings

settings = Settings.from_env()
db = settings.database

db_url = URL.create(
    drivername=db.drivername,
    username=db.username,
    password=db.password,
    host=db.host,
    port=db.port,
    database=db.database,
)

engine = create_engine(
    db_url,
    echo=False,
    # pool_size=5,
    # max_overflow=10,
    pool_pre_ping=True
)

def get_session():
    return Session(engine)
