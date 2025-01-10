import streamlit as st
from model import MultiLoRAViT
import torch
from PIL import Image
from torchvision import transforms
import json

# Set page config
st.set_page_config(page_title="Anime Character Attribute Predictor", layout="wide")

@st.cache_resource
def load_model():
    # Load encodings
    with open('encodings.json', 'r') as f:
        encodings = json.load(f)

    adapter_config = {
        'Age': len(encodings['Age']),
        'Gender': len(encodings['Gender']),
        'Ethnicity': len(encodings['Ethnicity']),
        'Hair Style': len(encodings['Hair Style']),
        'Hair Color': len(encodings['Hair Color']),
        'Hair Length': len(encodings['Hair Length']),
        'Eye Color': len(encodings['Eye Color']),
        'Body Type': len(encodings['Body Type']),
        'Dress': len(encodings['Dress'])
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiLoRAViT(adapter_config, r=4)
    epochs = model.load_model(exp='checkpoints')
    model = model.to(device)
    
    return model, encodings, device

def predict_attributes(image, model, encodings, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        predictions = {}
        
        with torch.no_grad():
            for adapter_name in encodings.keys():
                model.switch_adapter(adapter_name)
                output = model(image_tensor)
                pred_idx = output.argmax(1).item()
                
                reverse_encoding = {v: k for k, v in encodings[adapter_name].items()}
                predictions[adapter_name] = reverse_encoding[pred_idx]

        return predictions
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def main():
    st.title("Anime Character Attribute Predictor")
    
    # Load model and encodings
    with st.spinner("Loading model..."):
        model, encodings, device = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image and predictions side by side
        col1, col2 = st.columns(2)
        
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        col1.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make predictions
        with st.spinner("Analyzing image..."):
            predictions = predict_attributes(image, model, encodings, device)
        
        if predictions:
            # Display predictions in a nice format
            col2.subheader("Predicted Attributes:")
            for attribute, value in predictions.items():
                col2.write(f"**{attribute}:** {value}")

if __name__ == "__main__":
    main()