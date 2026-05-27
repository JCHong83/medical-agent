import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import ai_routes, appointment_routes

app = FastAPI(title="Medical AI Agent API")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Aggregating all routes
app.include_router(ai_routes.router)
app.include_router(appointment_routes.router)

@app.get("/health")
async def health():
  return {"status": "healthy"}

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)