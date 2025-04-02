![dfvbd.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/SaNubya9pBi7MhLhR8VVL.png)

# **Hand-Gesture-2-Robot**  

> **Hand-Gesture-2-Robot** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to recognize hand gestures and map them to specific robot commands using the **SiglipForImageClassification** architecture.  

```py
Classification Report:
                        precision    recall  f1-score   support

"rotate anticlockwise"     0.9926    0.9958    0.9942       944
            "increase"     0.9975    0.9975    0.9975       789
             "release"     0.9941    1.0000    0.9970       670
              "switch"     1.0000    0.9986    0.9993       728
             "look up"     0.9984    0.9984    0.9984       635
           "Terminate"     0.9983    1.0000    0.9991       580
            "decrease"     0.9942    1.0000    0.9971       684
       "move backward"     0.9986    0.9972    0.9979       725
               "point"     0.9965    0.9913    0.9939      1716
    "rotate clockwise"     1.0000    1.0000    1.0000       868
               "grasp"     0.9922    0.9961    0.9941       767
               "pause"     0.9991    1.0000    0.9995      1079
        "move forward"     1.0000    0.9944    0.9972       886
             "Confirm"     0.9983    0.9983    0.9983       573
           "look down"     0.9985    0.9970    0.9977       664
           "move left"     0.9952    0.9968    0.9960       622
          "move right"     1.0000    1.0000    1.0000       622

              accuracy                         0.9972     13552
             macro avg     0.9973    0.9977    0.9975     13552
          weighted avg     0.9972    0.9972    0.9972     13552
```

![download (2).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/d7PmqrxLjLfGqQCNmqnB3.png)

The model categorizes hand gestures into 17 different robot commands:  
- **Class 0:** "rotate anticlockwise"  
- **Class 1:** "increase"  
- **Class 2:** "release"  
- **Class 3:** "switch"  
- **Class 4:** "look up"  
- **Class 5:** "Terminate"  
- **Class 6:** "decrease"  
- **Class 7:** "move backward"  
- **Class 8:** "point"  
- **Class 9:** "rotate clockwise"  
- **Class 10:** "grasp"  
- **Class 11:** "pause"  
- **Class 12:** "move forward"  
- **Class 13:** "Confirm"  
- **Class 14:** "look down"  
- **Class 15:** "move left"  
- **Class 16:** "move right"  

# **Run with Transformers🤗**  

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```  

# **Intended Use:**  

The **Hand-Gesture-2-Robot** model is designed to classify hand gestures into corresponding robot commands. Potential use cases include:  

- **Human-Robot Interaction:** Enabling intuitive control of robots using hand gestures.  
- **Assistive Technology:** Helping individuals with disabilities communicate commands.  
- **Industrial Automation:** Enhancing robotic operations in manufacturing.  
- **Gaming & VR:** Providing gesture-based controls for immersive experiences.  
- **Security & Surveillance:** Implementing gesture-based access control.  
