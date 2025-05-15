import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Utility to enable dropout during inference
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# MC Dropout prediction function
def mc_dropout_predict(model, input_tensor, n_iter=30):
    model.eval()
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_iter):
            out = model(input_tensor).cpu().numpy().squeeze()
            preds.append(out)
    preds = np.stack(preds, axis=0)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

# ==== CONFIG ====
MODEL_PATH = "models/densenet121_xray.pt"
BEST_MODEL_PATH = "models/best_densenet.pt"  # New model path
IMAGE_SIZE = 224
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# ==== MODEL ====
class DenseNetXray(nn.Module):
    def __init__(self, num_labels=15):
        super().__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model(model_path=None):
    if model_path is None:
        model_path = MODEL_PATH
    model = DenseNetXray(num_labels=len(LABELS))
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        # Remove 'module.' prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        st.success("✅ Model loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
    model.eval()
    # Access the weight of the Linear layer (first layer in the Sequential)
    st.write("Model loaded, classifier weights mean:", model.model.classifier[0].weight.mean().item())
    return model

# ==== PREPROCESS ====
def preprocess(_image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])
    if _image.mode != 'RGB':
        _image = _image.convert('RGB')
    tensor = transform(_image).unsqueeze(0)  # Add batch dimension
    st.write("Preprocessed image shape:", tensor.shape)
    st.write("Preprocessed image range:", tensor.min().item(), "to", tensor.max().item())
    return tensor

# ==== GRAD-CAM UTILS ====
def generate_gradcam(model, input_tensor, target_class, target_layer='features.denseblock4'):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    layer = dict([*model.model.named_modules()])[target_layer]
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()

    # Get hooked gradients and activations
    grads_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]

    # Compute weights and Grad-CAM
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations_val[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    if cam.max() != 0:
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
    else:
        cam = np.zeros_like(cam)
    
    # Clean up hooks
    forward_handle.remove()
    backward_handle.remove()
    return cam

def overlay_heatmap(img: Image.Image, cam: np.ndarray, alpha=0.4):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
    img_np = np.array(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ==== UI ====
st.set_page_config(page_title="Chest X-ray Classifier", layout="centered")
st.title("🩻 Chest X-ray Disease Classifier")

st.write("Upload a chest X-ray and get prediction probabilities for 15 common conditions.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    
    # Print image info
    st.write("Original image size:", image.size)
    st.write("Original image mode:", image.mode)

    if np.array(image).std() < 5:
        st.warning("This image appears to be blank or nearly blank. Results will not be meaningful.")

    # Model selection
    model_option = st.radio("Select Model", ["best_densenet.pt", "densenet121_xray.pt"])
    model_path = BEST_MODEL_PATH if model_option == "best_densenet.pt" else MODEL_PATH
    model = load_model(model_path)
    input_tensor = preprocess(image)

    # MC Dropout Uncertainty Estimation
    st.write("Running MC Dropout for uncertainty estimation...")
    probs_mean, probs_std = mc_dropout_predict(model, input_tensor, n_iter=30)
    st.write("Raw model output (mean):", probs_mean)
    st.write("Raw model output (std):", probs_std)

    st.subheader("📊 Prediction Probabilities:")
    for label, mean, std in sorted(zip(LABELS, probs_mean, probs_std), key=lambda x: -x[1]):
        st.write(f"**{label}**: {mean:.3f} ± {std:.3f}")

    # === Grad-CAM Visualization ===
    st.subheader("🩻 Grad-CAM Visualization")
    selected_label = st.selectbox("Select a disease to visualize Grad-CAM:", LABELS)
    selected_idx = LABELS.index(selected_label)
    if st.button("Show Grad-CAM Heatmap"):
        with st.spinner("Generating Grad-CAM..."):
            cam = generate_gradcam(model, input_tensor, selected_idx)
            overlay = overlay_heatmap(image, cam)
            st.image(overlay, caption=f"Grad-CAM for {selected_label}", use_container_width=True) 