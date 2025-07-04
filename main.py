import torch
from torchvision import transforms
from PIL import Image
from utils import get_user_test_data, get_precautions_from_gemini
from models.TumorModel import TumorClassification, GliomaStageModel

# Load tumor type prediction model
tumor_model = TumorClassification()
tumor_model.load_state_dict(torch.load('models/BTD_model.pth', map_location=torch.device('cpu')))
tumor_model.eval()

# Load glioma stage model
glioma_model = GliomaStageModel()
glioma_model.load_state_dict(torch.load('models/glioma_stages.pth'))
glioma_model.eval()

def predict_tumor(image_path):

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = tumor_model(image)
        predicted = torch.argmax(outputs, dim=1).item()
        labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        # return labels[predicted]
        return labels[predicted]


def classify_glioma_stage(test_data):
    input_tensor = torch.tensor(test_data).float().unsqueeze(0)
    with torch.no_grad():
        output = glioma_model(input_tensor)
        _, predicted = torch.max(output, 1)
        stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
        return stages[predicted.item()]

def main_pipeline(image_path):
    tumor_type = predict_tumor(image_path)
    print(f"Predicted Tumor Type: {tumor_type}")

    if tumor_type == "glioma":
        print("Glioma detected. Please enter additional test results:")
        test_data = get_user_test_data()
        stage = classify_glioma_stage(test_data)
        print(f"Glioma Stage: {stage}")
    else:
        precaution = get_precautions_from_gemini(tumor_type)
        print(f"Precaution for {tumor_type}: {precaution}")

if __name__ == "__main__":
    main_pipeline("images/glioma (1).jpg")
