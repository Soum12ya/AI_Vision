from fastapi import FastAPI
from .api.routes import router as blueprints_router

app = FastAPI(title="AI Vision Takeoff")
app.include_router(blueprints_router)

@app.get("/")
def root():
    return {"status":"ok","message":"AI Vision Takeoff API"}