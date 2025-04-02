import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Hand-Gesture-2-Robot"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def gesture_classification(image):
    """Predicts the robot command from a hand gesture image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "rotate anticlockwise", 
        "1": "increase", 
        "2": "release", 
        "3": "switch", 
        "4": "look up", 
        "5": "Terminate", 
        "6": "decrease", 
        "7": "move backward", 
        "8": "point", 
        "9": "rotate clockwise", 
        "10": "grasp", 
        "11": "pause", 
        "12": "move forward", 
        "13": "Confirm", 
        "14": "look down", 
        "15": "move left", 
        "16": "move right"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=gesture_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Hand Gesture to Robot Command",
    description="Upload an image of a hand gesture to predict the corresponding robot command."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
