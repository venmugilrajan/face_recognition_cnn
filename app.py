import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

class FER_CNNModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FER_CNNModel, self).__init__()
        
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding='same')
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.25)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.drop_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.drop_fc2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.drop4(x)
        x = self.flatten(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop_fc2(x)
        x = self.fc3(x)
        return x

MODEL_PATH = 'fer_model.pth'
NUM_CLASSES = 7
IMG_SIZE = (48, 48)

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(face_cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found at {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        print("Cascade classifier failed to load. Using fallback.")
        raise Exception("Cascade load error")
except Exception as e:
    print(f"Error loading Haar cascade: {e}")
    print("Please ensure OpenCV is installed correctly.")
    class DummyCascade:
        def detectMultiScale(self, *args, **kwargs):
            return []
    face_cascade = DummyCascade()

try:
    model = FER_CNNModel(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please make sure 'fer_model.pth' is in the same folder as app.py")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

preprocess_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

def predict(image_np):
    if image_np is None:
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank_image, {"No Image": 1.0}
        
    try:
        gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    except cv2.error as e:
        return image_np, {f"Error": 1.0}

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

    if len(faces) == 0:
        return image_np, { "No Face Detected": 1.0 }

    (x, y, w, h) = faces[0]
    
    cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

    face_roi = gray_img[y:y+h, x:x+w]
    
    pil_img = Image.fromarray(face_roi)

    tensor_img = preprocess_transform(pil_img)
    
    tensor_img = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probs = F.softmax(outputs, dim=1)
    
    prob_list = probs.cpu().numpy().flatten()
    output_dict = {CLASS_NAMES[i]: float(prob_list[i]) for i in range(NUM_CLASSES)}
    
    return image_np, output_dict

title = "Facial Emotion Recognition (PyTorch)"
description = (
    "Use your webcam to detect an emotion. "
    "A green box will show the detected face. Ensure your face is well-lit and facing the camera. "
    "Based on the FER2013 dataset."
)

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        label="Input Image",
        sources=["webcam"],
        type="numpy",
        mirror_webcam=True
    ),
    outputs=[
        gr.Image(label="Webcam Feed", type="numpy"),
        gr.Label(num_top_classes=3, label="Top 3 Emotions")
    ],
    title=title,
    description=description,
    live=True
)

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: '{MODEL_PATH}' not found.")
        print("Please run your training script first to create the model file.")
    else:
        print(f"Launching Gradio app... (Model: {MODEL_PATH})")
        iface.launch()
