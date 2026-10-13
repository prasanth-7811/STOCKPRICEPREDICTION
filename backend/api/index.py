"""
backend/api/index.py — Backend service entrypoint for Vercel.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI(title="Stock Prediction Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "backend", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


handler = Mangum(app)
