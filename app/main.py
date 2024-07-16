import uvicorn
from fastapi import FastAPI
from endpoints import routes
import os

app = FastAPI()

app.include_router(routes.router, prefix="/api", tags=["api"])

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    cert_file = os.path.join(os.path.dirname(__file__), '..', 'cert.pem')
    key_file = os.path.join(os.path.dirname(__file__), '..', 'key.pem')
    
    uvicorn.run(app, host="0.0.0.0", port=443, ssl_keyfile=key_file, ssl_certfile=cert_file)
