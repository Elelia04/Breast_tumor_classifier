import sys
import os
import gdown
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

import streamlit as st
import base64
from PIL import Image
import torch
from torchvision import transforms
from model import get_resnet50 
import torch.nn.functional as F
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Grad-CAM

import cv2
import numpy as np

def generate_gradcam(model, img_tensor, target_layer):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax().item()
    score = output[:, pred_class]
    model.zero_grad()
    score.backward()

    hook_f.remove()
    hook_b.remove()

    grad = gradients[0]
    act = activations[0]

    pooled_grad = grad.mean(dim=(2, 3), keepdim=True)
    weighted_act = pooled_grad * act
    cam = weighted_act.sum(dim=1).squeeze()

    cam = cam.cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam
    cam = cv2.resize(cam, (224, 224))
    cam = np.uint8(255 * cam)
    return cam, pred_class


#INTERFACE 

#Uploading the image 
st.title("MILF model: upoload and analyze")

magnification = st.selectbox("Select the image magnification:", ["40X", "100X", "200X", "400X"])

#Imported from drive
MODEL_URLS = {
    "40X": "https://drive.google.com/uc?id=1BxyE2coxsy_CZJCmmEy4IFOiWfZF1B_e",
    "100X": "https://drive.google.com/uc?id=1tyjYomweRk85d0q1pRIz0rtLfD6d423H",
    "200X": "https://drive.google.com/uc?id=1f5aibGwtXV37VgQ1Ou_xgFMLn3f-5WUO",
    "400X": "https://drive.google.com/uc?id=1FgDvNAB8G1dyOuaFc_51z99jXvHkNYr7"
}

MODEL_PATHS = {
    "40X": "model_40X_b64_n500.pt",
    "100X": "model_100X_b64_n600.pt",
    "200X": "model_200X_b32_n500.pt",
    "400X": "model_400X_b32_n500.pt"
}

conf_matrix_paths = {
    "40X": "model_40X_b64_n500_CM.png",
    "100X": "model_100X_b64_n600_CM.png",
    "200X": "model_200X_b32_n500_CM.png",
    "400X": "model_400X_b32_n500_CM.png"
}
cm_path = conf_matrix_paths[magnification]

def download_model(mag):
    path = MODEL_PATHS[mag]
    url = MODEL_URLS[mag]
    if not os.path.exists(path):
        with st.spinner(f"Downloading model {mag}..."):
            gdown.download(url, path, quiet=False)
    return path


model_path = download_model(magnification)
uploaded_file = st.file_uploader("Upload a microscopic breast tumor image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    #Image preprocessing 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)  

    #Model
    model = get_resnet50(num_classes=2, fine_tune_layers=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    #Prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, 1).item()

        label = "Malignant" if prediction == 1 else "Benign"
        color = "d9534f" if prediction == 1 else "5cb85c"

        st.markdown(f"""
            <div style='
                background-color: {color};
                color: white;
                padding: 15px 20px;
                border-radius: 15px;
                text-align: center;
                width: fit-content;
                margin: auto;
                opacity: 0.92;
                font-size: 18px;
                font-family: Helvetica;
                font-weight: bold;
                box-shadow: 0 0 12px rgba(0,0,0,0.15);
            '>
                Prediction: <strong>{label}</strong>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)


    #GRAD-CAM IMPLEMENTATION, one image at a time
        
    target_layer = model.layer4[1].conv2  

    col1, col2 = st.columns(2)

    cam, pred_class = generate_gradcam(model, input_tensor, target_layer)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    image_np = np.array(image.resize((224, 224)))
    if image_np.max() <= 1:
        image_np = (image_np * 255).astype(np.uint8)

    superimposed = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    with col1:
        st.image(superimposed, width=320)
        st.markdown("<div style='text-align: center; font-weight: bold; margin-top: 5px;'>Grad-CAM</div>", unsafe_allow_html=True)
    with col2:
        st.image(cm_path, width=400)
        st.markdown("<div style='text-align: center; font-weight: bold; margin-top: 5px;'>Confusion Matrix</div>", unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)

    
    #ACCURACY VALUES 

    col1, col2, col3 = st.columns(3)

    metrics_by_mag = {
    "40X": {"Accuracy<br>Train": "99.7%", "Accuracy<br>Val": "86.7%", "Magnification": "40X"},
    "100X": {"Accuracy<br>Train": "96.9%", "Accuracy<br>Val": "94.4%", "Magnification": "100X"},
    "200X": {"Accuracy<br>Train": "98.9%", "Accuracy<br>Val": "79.5%", "Magnification": "200X"},
    "400X": {"Accuracy<br>Train": "98.2%", "Accuracy<br>Val": "85.5%", "Magnification": "400X"},
    }

    metrics = metrics_by_mag[magnification]


    for i, (title, value) in enumerate(metrics.items()):
        col = [col1, col2, col3][i]
        with col:
            st.markdown(f"""
            <div style="
                background-color: rgba(255, 255, 255, 0.07);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 0 8px rgba(255,255,255,0.2);
                font-family: Helvetica;
                min-height: 85px;
            ">
                <div style='font-size: 16px; font-weight: bold; margin-bottom: 8px; line-height: 1.4;'>{title}</div>
                <p style='font-size: 22px; margin: 0;'>{value}</p>
            </div>
            """, unsafe_allow_html=True)



#SIDEBAR COLOUR
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.6);  
        backdrop-filter: blur(4px);           
        border-right: 1px solid rgba(255,255,255,0.1);
        color: white;
    }

    [data-testid="stSidebarNav"] > ul {
        padding-top: 30px;
    }

    [data-testid="stSidebarNav"] > ul > li {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        padding: 8px;
    }

    [data-testid="stSidebarNav"] > ul > li:hover {
        background-color: rgba(0,255,170,0.15);
        border-radius: 8px;
        transition: 0.3s ease-in-out;
    }
    </style>
""", unsafe_allow_html=True)


#BACKROUND IMAGE

def set_background(image_file_name):
    current_dir = os.path.dirname(os.path.dirname(__file__))
    image_path = os.path.join(current_dir, image_file_name)
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Poi chiama:
set_background("abstract-digital-grid-black-background.jpg")

