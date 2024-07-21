from fastapi import FastAPI
from signconnection.api.endpoints import predict, root

app = FastAPI()

# app.add_api_route("/predict", predict, methods=["POST"])
app.add_api_route("/predict", predict, methods=["GET"])
app.add_api_route("/", root, methods=["GET"])
