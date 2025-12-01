"""
VeriPix Classifier GUI - FINAL WORKING VERSION
Uses EXACT same preprocessing as test_classifier_quick.py
"""

import gradio as gr
import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_classifier import create_efficientnet_classifier
from preprocessing.ela import ELAProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VeriPixGUI:
    """Working GUI with exact same preprocessing as test script"""
    
    def __init__(self):
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = 224
        
        # Initialize ELA with EXACT same parameters as test script
        self.ela_processor = ELAProcessor(quality=90, scale=10)
        
        # Initialize transform with EXACT same parameters as test script
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Load model
        print("Loading classifier...")
        self.classifier = create_efficientnet_classifier(pretrained=False, device=self.device)
        checkpoint = torch.load('checkpoints/classifier/best.pth', map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        print(f"✓ Classifier loaded (F1: {checkpoint.get('best_f1', 0):.4f})")
    
    def predict_from_path(self, image_path, threshold=0.5):
        """
        Predict from file path - EXACT same as test script
        """
        if not Path(image_path).exists():
            return "❌ File not found", 0.0, None
        
        try:
            # Load and preprocess - EXACT same as test script
            ela_img = self.ela_processor.apply_ela(image_path, output_size=(224, 224))
            transformed = self.transform(image=ela_img)
            tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Predict - EXACT same as test script
            with torch.no_grad():
                logits = self.classifier(tensor)
                prob = torch.sigmoid(logits).item()
            
            # Determine label
            is_tampered = prob >= threshold
            label = "🔴 TAMPERED" if is_tampered else "✅ AUTHENTIC"
            confidence_pct = prob * 100
            
            # Debug
            print(f"[Debug] File: {Path(image_path).name}")
            print(f"[Debug] Logit: {logits.item():.4f}, Prob: {prob:.4f}, Label: {label}")
            
            # Load original for visualization
            original = cv2.imread(str(image_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            original = cv2.resize(original, (224, 224))
            
            # Add border
            border_color = (255, 0, 0) if is_tampered else (0, 255, 0)
            visualization = cv2.copyMakeBorder(
                original, 15, 15, 15, 15,
                cv2.BORDER_CONSTANT, 
                value=border_color
            )
            
            return label, confidence_pct, visualization
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", 0.0, None
    
    def predict_from_upload(self, image_array, threshold=0.5):
        """
        Predict from uploaded numpy array
        More robust handling of different image formats
        """
        if image_array is None:
            return "❌ No image uploaded", 0.0, None
        
        try:
            # Normalize image array
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                # Convert float [0,1] to uint8 [0,255]
                image_array = (image_array * 255).astype(np.uint8)
            
            # Ensure uint8
            image_array = image_array.astype(np.uint8)
            
            # Handle grayscale
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            # Handle RGBA
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            # Ensure RGB (Gradio gives RGB, but PIL needs it too)
            elif image_array.shape[2] == 3:
                pass  # Already RGB
            
            # Save to temp file for ELA processing
            import tempfile
            import os
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', dir='.')
            os.close(temp_fd)
            
            try:
                # Save as JPEG using PIL (ensures correct format)
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(image_array, mode='RGB')
                pil_img.save(temp_path, 'JPEG', quality=95)
                
                # Use same prediction as file path
                result = self.predict_from_path(temp_path, threshold)
                
                return result
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", 0.0, None



def create_app():
    """Create Gradio interface"""
    
    gui = VeriPixGUI()
    
    with gr.Blocks(title="VeriPix - Working Classifier") as app:
        gr.Markdown("""
        # 🔍 VeriPix - Image Forgery Classifier (WORKING VERSION)
        
        **Status:** ✅ Using exact same preprocessing as test script
        
        **Performance:** F1-Score 88.77% | Accuracy 89.3% | AUC-ROC 0.97
        """)
        
        with gr.Tabs():
            # Tab 1: Upload Image
            with gr.Tab("📤 Upload Image"):
                with gr.Row():
                    with gr.Column():
                        upload_input = gr.Image(label="Upload Image", type="numpy")
                        upload_threshold = gr.Slider(0, 1, 0.5, 0.05, label="Threshold")
                        upload_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    with gr.Column():
                        upload_label = gr.Textbox(label="Prediction", lines=1)
                        upload_conf = gr.Number(label="Confidence (%)", precision=2)
                        upload_vis = gr.Image(label="Result")
                
                upload_btn.click(
                    fn=gui.predict_from_upload,
                    inputs=[upload_input, upload_threshold],
                    outputs=[upload_label, upload_conf, upload_vis]
                )
            
            # Tab 2: Test Examples
            with gr.Tab("📋 Test Examples"):
                gr.Markdown("**Click buttons to test on known images:**")
                
                example_threshold = gr.Slider(0, 1, 0.5, 0.05, label="Threshold")
                
                with gr.Row():
                    btn1 = gr.Button("🔴 Tampered Example 1", variant="secondary")
                    btn2 = gr.Button("🔴 Tampered Example 2", variant="secondary")
                    btn3 = gr.Button("✅ Authentic Example", variant="secondary")
                
                example_label = gr.Textbox(label="Prediction", lines=1)
                example_conf = gr.Number(label="Confidence (%)", precision=2)
                example_vis = gr.Image(label="Result")
                
                # Connect buttons
                btn1.click(
                    fn=lambda t: gui.predict_from_path(
                        "data/processed/test/tampered/casia_Tp_D_CNN_M_N_art00052_arc00030_11853.jpg", t
                    ),
                    inputs=[example_threshold],
                    outputs=[example_label, example_conf, example_vis]
                )
                
                btn2.click(
                    fn=lambda t: gui.predict_from_path(
                        "data/processed/test/tampered/casia_Tp_D_CND_M_N_art00077_art00076_10290.tif", t
                    ),
                    inputs=[example_threshold],
                    outputs=[example_label, example_conf, example_vis]
                )
                
                btn3.click(
                    fn=lambda t: gui.predict_from_path(
                        "data/processed/test/authentic/casia_Au_ani_00004.jpg", t
                    ),
                    inputs=[example_threshold],
                    outputs=[example_label, example_conf, example_vis]
                )
        
        gr.Markdown("""
        ---
        ### 📊 Model Info
        - **Architecture:** EfficientNet-B0 (4.7M params)
        - **Preprocessing:** ELA (quality=90, scale=10)
        - **Training:** 49 epochs, early stopping
        - **Datasets:** CASIA v2.0 + CoMoFoD (19,014 images)
        """)
    
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7863)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )
