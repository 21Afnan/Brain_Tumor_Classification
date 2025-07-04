from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
from torchvision import transforms
from PIL import Image
import io

from models.TumorModel import TumorClassification, GliomaStageModel
from utils import get_precautions_from_gemini

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
tumor_model = TumorClassification()
tumor_model.load_state_dict(torch.load('models/BTD_model.pth', map_location=torch.device('cpu')))
tumor_model.eval()

glioma_model = GliomaStageModel()
glioma_model.load_state_dict(torch.load('models/glioma_stages.pth', map_location=torch.device('cpu')))
glioma_model.eval()

# Transform for image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Label list
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = tumor_model(image)
        predicted = torch.argmax(outputs, dim=1).item()
        tumor_type = labels[predicted]

    if tumor_type == "glioma":
        return {"tumor_type": tumor_type, "next": "submit_mutation_data"}
    else:
        precaution = get_precautions_from_gemini(tumor_type)
        return {"tumor_type": tumor_type, "precaution": precaution}

class MutationInput(BaseModel):
    gender: str  # 'm' or 'f'
    age: float
    idh1: int
    tp53: int
    atrx: int
    pten: int
    egfr: int
    cic: int
    pik3ca: int

@app.post("/predict-glioma-stage")
def predict_glioma_stage(data: MutationInput):
    gender = 0 if data.gender.lower() == 'm' else 1
    test_data = [
        gender, data.age, data.idh1, data.tp53, data.atrx,
        data.pten, data.egfr, data.cic, data.pik3ca
    ]

    input_tensor = torch.tensor(test_data).float().unsqueeze(0)
    with torch.no_grad():
        output = glioma_model(input_tensor)
        _, predicted = torch.max(output, 1)
        stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
        stage = stages[predicted.item()]

    return {"glioma_stage": stage}

@app.get("/")
def root():
    return {"message": "Brain Tumor Detection API is running."}
