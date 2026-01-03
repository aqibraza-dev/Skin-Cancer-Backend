from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from app.pipeline import classifier

app = FastAPI(title="Skin Cancer Detection API")

# CORS - Allow React Frontend to connect
origins = [
    "http://localhost:3000",
    "http://localhost:5173",  # Vite default
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (images for the frontend sample)
# Ensure you create a folder named 'assets' in the root backend directory
os.makedirs("assets", exist_ok=True)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

@app.get("/")
def home():
    return {"message": "Skin Cancer Detection API is running"}

@app.post("/predict")
async def predict_skin_lesion(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the predicted skin cancer class.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        content = await file.read()
        result = classifier.predict(content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)