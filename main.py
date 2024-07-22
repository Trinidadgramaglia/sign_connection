from fastapi import FastAPI
from signconnection.api.endpoints import app as endpoints_app

app = FastAPI()

app.mount("/", endpoints_app)
