import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNModel  

# class labels
classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# load model
model = CNNModel()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

def predict(image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

        probs = torch.softmax(output, dim=1)

        # top 3 predictions
        top_probs, top_idxs = torch.topk(probs, 3)

        results = {}
        for i in range(3):
            label = classes[top_idxs[0][i].item()]
            confidence = top_probs[0][i].item()
            results[label] = round(confidence, 3)

    return results

# UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Fashion MNIST Classifier",
    description="Upload an image of clothing to classify"
)

interface.launch()