from fastapi import FastAPI
from healthcare.routes import router as healthcare_router

app = FastAPI(title="Decentralized AI Healthcare")
app.include_router(healthcare_router)

@app.get("/")
def root():
    return {"message": "Welcome to the Decentralized AI Healthcare platform powered by Bittensor!"}
