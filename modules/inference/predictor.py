import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

from ..model_training.model import create_model

class LungDiseasePredictor:
    def __init__(self, model_path='artifacts/models/trained_model.pth') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found.")
        
        model = create_model(num_classes=1)
        model.load_state_dict(torch.load('artifacts/models/trained_model.pth'))
        model.to(self.device)
        model.eval()
        return model

    def _get_transform(self, size=224):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        input_tensor = image.unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()

        # Interpret the result
        prediction = "Disease detected" if probability > 0.5 else "No disease detected"
        confidence = probability if prediction == "Disease detected" else 1 - probability

        return {
            "prediction": prediction,
            "confidence": f"{confidence:.2%}",
            "raw_probability": probability
        }

# Example usage
if __name__ == "__main__":
    predictor = LungDiseasePredictor()

    sample_img_name = "00018387_036.png"
    img_dir = "D:\BigData\images_009\images"
    sample_img_path = os.path.join(img_dir, sample_img_name)

    result = predictor.predict(sample_img_path)
    print(result)