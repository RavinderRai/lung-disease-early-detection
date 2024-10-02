import gradio as gr
import os
import random

from ..inference.predictor import LungDiseasePredictor

predictor = LungDiseasePredictor()

def detect_which_disease_placeholder():
    """
    This function randomly selects a disease from a list of possible lung diseases.
    It is a placeholder function for another model to be trained on only images with diseases,
    where we predict which disease is present.
    """
    possible_diseases = [
        'Infiltration',
        'Effusion',
        'Atelectasis',
        'Nodule',
        'Mass',
        'Pneumothorax',
        'Consolidation',
        'Pleural_Thickening',
        'Cardiomegaly',
        'Emphysema',
        'Edema',
        'Fibrosis',
        'Pneumonia',
        'Hernia'
    ]
    return random.choice(possible_diseases)

def predict_lung_disease(image):
    """
    Predicts lung disease from a given image and returns the image along with the prediction result.

    Parameters:
    image (PIL.Image): The input image to be analyzed for lung disease detection.

    Returns:
    tuple: A tuple containing the original image and a string describing the prediction result.
    """
    temp_path = "temp_path.png"
    image.save(temp_path)

    result = predictor.predict(temp_path)

    os.remove(temp_path)

    prediction = result['prediction']
    confidence = result['confidence']

    if prediction == "Disease Detected":
        disease = detect_which_disease_placeholder()
        predicted_disease = f"The disease {disease} was detected with {confidence} certainty."
    else:
        predicted_disease = f"No disease was detected with {confidence} certainty."
    
    return image, predicted_disease



def main():
    description = """
    ### Upload a chest X-ray image, and the model will predict whether a disease is detected. If a disease is detected, the app will also suggest which disease it might be.
    """

    # Create the Gradio interface
    interface = gr.Interface(
        fn=predict_lung_disease,           
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=[
            gr.Image(type="pil", label="Uploaded Image", visible=False),  # Hidden output for the image, otherwise you'll see duplicate displays
            gr.Textbox(label="Prediction Result")  
        ],
        title="Lung Disease Detection",
        description=description,
        theme="compact",
        allow_flagging="never"
    )

    interface.launch()

if __name__ == "__main__":
    main()