from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from app.pipeline import classifier

app = FastAPI(title="Skin Cancer Detection API")

# CORS - Update this list!
origins = [
    "http://localhost:3000",
    "http://localhost:5173", 
    "https://med-ai-pro.vercel.app",  # <--- CRITICAL: Your Vercel domain
    "*"                               # Keep '*' only for testing, remove in strict prod
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("assets", exist_ok=True)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

@app.get("/")
def home():
    return {
        "message": "Skin Cancer Detection API is running",
        "status": "active"
    }

@app.post("/predict")
async def predict_skin_lesion(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        content = await file.read()
        result = classifier.predict(content)
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Use 'app' object directly for cleaner script execution
    uvicorn.run(app, host="0.0.0.0", port=8000)
