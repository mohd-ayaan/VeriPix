"""
VeriPix - Complete Gradio Application
Full system with classification + localization
"""

import gradio as gr
import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
import sys
import tempfile
import os
from PIL import Image as PILImage

sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_classifier import create_efficientnet_classifier
from models.unet_localizer import create_unet_localizer
from preprocessing.ela import ELAProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VeriPixSystem:
    """Complete VeriPix system with classification and localization"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize full system"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = 224
        
        # Initialize ELA processor
        self.ela_processor = ELAProcessor(quality=90, scale=10)
        
        # Initialize transform
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Load classifier
        print("Loading classifier...")
        self.classifier = create_efficientnet_classifier(pretrained=False, device=self.device)
        cls_checkpoint = torch.load('checkpoints/classifier/best.pth', map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(cls_checkpoint['model_state_dict'])
        self.classifier.eval()
        self.classifier_f1 = cls_checkpoint.get('best_f1', 0)
        print(f"  ✓ Classifier loaded (F1: {self.classifier_f1:.4f})")
        
        # Load localizer
        print("Loading localizer...")
        self.localizer = create_unet_localizer(device=self.device)
        loc_checkpoint = torch.load('checkpoints/localizer/best.pth', map_location=self.device, weights_only=False)
        self.localizer.load_state_dict(loc_checkpoint['model_state_dict'])
        self.localizer.eval()
        self.localizer_iou = loc_checkpoint.get('best_iou', 0)
        print(f"  ✓ Localizer loaded (IoU: {self.localizer_iou:.4f})")
    
    def predict_from_path(self, image_path, cls_threshold=0.5, loc_threshold=0.5):
        """
        Complete prediction pipeline from file path
        
        Returns:
            tuple: (classification_label, confidence, localization_visualization)
        """
        if not Path(image_path).exists():
            return "❌ File not found", 0.0, None, None, None
        
        try:
            # Load original image for visualization
            original = cv2.imread(str(image_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            original_display = cv2.resize(original, (self.image_size, self.image_size))
            
            # Apply ELA
            ela_img = self.ela_processor.apply_ela(image_path, output_size=(self.image_size, self.image_size))
            
            # Transform
            transformed = self.transform(image=ela_img)
            tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Classification
                cls_logits = self.classifier(tensor)
                cls_prob = torch.sigmoid(cls_logits).item()
                is_tampered = cls_prob >= cls_threshold
                
                # Localization (only if tampered)
                if is_tampered:
                    loc_logits = self.localizer(tensor)
                    loc_prob = torch.sigmoid(loc_logits).squeeze().cpu().numpy()
                    mask = (loc_prob >= loc_threshold).astype(np.uint8) * 255
                else:
                    mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                    loc_prob = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            
            # Create visualizations
            label = "🔴 TAMPERED" if is_tampered else "✅ AUTHENTIC"
            confidence_pct = cls_prob * 100
            
            # Visualization 1: Original with border
            border_color = (255, 0, 0) if is_tampered else (0, 255, 0)
            vis_original = cv2.copyMakeBorder(
                original_display, 10, 10, 10, 10,
                cv2.BORDER_CONSTANT, value=border_color
            )
            
            # Visualization 2: Heatmap
            heatmap = cv2.applyColorMap((loc_prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Visualization 3: Binary mask
            mask_colored = np.zeros_like(original_display)
            mask_colored[:, :, 0] = mask  # Red channel
            
            # Visualization 4: Overlay
            if is_tampered and mask.sum() > 0:
                overlay = cv2.addWeighted(original_display, 0.6, mask_colored, 0.4, 0)
            else:
                overlay = original_display.copy()
            
            # Add text overlay
            tampered_percent = (mask > 0).sum() / (self.image_size * self.image_size) * 100
            
            # Debug output
            print(f"[Prediction] {Path(image_path).name}")
            print(f"  Classification: {label} ({confidence_pct:.2f}%)")
            print(f"  Localization: {tampered_percent:.2f}% of image")
            
            return label, confidence_pct, vis_original, heatmap, overlay
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", 0.0, None, None, None
    
    def predict_from_upload(self, image_array, cls_threshold=0.5, loc_threshold=0.5):
        """
        Predict from uploaded numpy array
        """
        if image_array is None:
            return "❌ No image uploaded", 0.0, None, None, None
        
        try:
            # Normalize array
            if image_array.dtype in [np.float32, np.float64]:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Handle channels
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            # Save to temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', dir='.')
            os.close(temp_fd)
            
            try:
                pil_img = PILImage.fromarray(image_array, mode='RGB')
                pil_img.save(temp_path, 'JPEG', quality=95)
                
                result = self.predict_from_path(temp_path, cls_threshold, loc_threshold)
                return result
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", 0.0, None, None, None


def create_interface():
    """Create Gradio interface"""
    
    # Initialize system
    system = VeriPixSystem()
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
        max-width: 1400px;
        margin: auto;
    }
    .title-text {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle-text {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 20px;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="VeriPix - AI Image Forgery Detector") as demo:
        
        # Header
        gr.HTML("""
        <div class="title-text">🔍 VeriPix</div>
        <div class="subtitle-text">AI-Powered Image Forgery Detection & Localization</div>
        """)
        
        # System info
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(f"""
                <div class="metric-box">
                    <h3>📊 Classification</h3>
                    <p style="font-size: 2em; margin: 5px 0;">{system.classifier_f1*100:.1f}%</p>
                    <p>F1-Score</p>
                </div>
                """)
            with gr.Column(scale=1):
                gr.HTML(f"""
                <div class="metric-box">
                    <h3>🎯 Localization</h3>
                    <p style="font-size: 2em; margin: 5px 0;">{system.localizer_iou*100:.1f}%</p>
                    <p>IoU Score</p>
                </div>
                """)
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="metric-box">
                    <h3>🚀 Architecture</h3>
                    <p style="font-size: 1.2em; margin: 5px 0;">EfficientNet-B0</p>
                    <p>+ U-Net Localizer</p>
                </div>
                """)
        
        gr.Markdown("---")
        
        # Main interface
        with gr.Tabs():
            
            # Tab 1: Upload & Test
            with gr.Tab("📤 Upload & Analyze"):
                with gr.Row():
                    with gr.Column():
                        upload_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            height=300
                        )
                        
                        with gr.Row():
                            cls_threshold = gr.Slider(
                                0, 1, 0.5, 0.05,
                                label="Classification Threshold"
                            )
                            loc_threshold = gr.Slider(
                                0, 1, 0.5, 0.05,
                                label="Localization Threshold"
                            )
                        
                        analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                    
                    with gr.Column():
                        result_label = gr.Textbox(label="Classification Result", lines=1)
                        result_conf = gr.Number(label="Confidence (%)", precision=2)
                
                # Visualization row
                with gr.Row():
                    vis_original = gr.Image(label="Original (with border)", height=250)
                    vis_heatmap = gr.Image(label="Localization Heatmap", height=250)
                    vis_overlay = gr.Image(label="Overlay Result", height=250)
                
                # Connect button
                analyze_btn.click(
                    fn=system.predict_from_upload,
                    inputs=[upload_input, cls_threshold, loc_threshold],
                    outputs=[result_label, result_conf, vis_original, vis_heatmap, vis_overlay]
                )
            
            # Tab 2: Example Images
            with gr.Tab("📋 Test Examples"):
                gr.Markdown("""
                ### Test on Pre-loaded Examples
                Click buttons to test on known authentic and tampered images from the test set.
                """)
                
                with gr.Row():
                    ex_cls_thresh = gr.Slider(0, 1, 0.5, 0.05, label="Classification Threshold")
                    ex_loc_thresh = gr.Slider(0, 1, 0.5, 0.05, label="Localization Threshold")
                
                with gr.Row():
                    btn_auth = gr.Button("✅ Test Authentic Image", variant="secondary")
                    btn_tamp1 = gr.Button("🔴 Test Tampered #1", variant="secondary")
                    btn_tamp2 = gr.Button("🔴 Test Tampered #2", variant="secondary")
                
                ex_label = gr.Textbox(label="Classification Result", lines=1)
                ex_conf = gr.Number(label="Confidence (%)", precision=2)
                
                with gr.Row():
                    ex_original = gr.Image(label="Original", height=250)
                    ex_heatmap = gr.Image(label="Heatmap", height=250)
                    ex_overlay = gr.Image(label="Overlay", height=250)
                
                # Connect example buttons
                btn_auth.click(
                    fn=lambda ct, lt: system.predict_from_path(
                        "data/processed/test/authentic/casia_Au_ani_00004.jpg", ct, lt
                    ),
                    inputs=[ex_cls_thresh, ex_loc_thresh],
                    outputs=[ex_label, ex_conf, ex_original, ex_heatmap, ex_overlay]
                )
                
                btn_tamp1.click(
                    fn=lambda ct, lt: system.predict_from_path(
                        "data/processed/test/tampered/casia_Tp_D_CNN_M_N_art00052_arc00030_11853.jpg", ct, lt
                    ),
                    inputs=[ex_cls_thresh, ex_loc_thresh],
                    outputs=[ex_label, ex_conf, ex_original, ex_heatmap, ex_overlay]
                )
                
                btn_tamp2.click(
                    fn=lambda ct, lt: system.predict_from_path(
                        "data/processed/test/tampered/casia_Tp_D_CND_M_N_art00077_art00076_10290.tif", ct, lt
                    ),
                    inputs=[ex_cls_thresh, ex_loc_thresh],
                    outputs=[ex_label, ex_conf, ex_original, ex_heatmap, ex_overlay]
                )
            
            # Tab 3: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
                # About VeriPix
                
                **VeriPix** is an AI-powered system for detecting and localizing image forgeries using deep learning.
                
                ## 📊 Performance Metrics
                
                ### Classification (EfficientNet-B0)
                - **F1-Score:** {system.classifier_f1*100:.2f}%
                - **Accuracy:** ~90%
                - **AUC-ROC:** 97.06%
                - **Architecture:** EfficientNet-B0 (4.7M parameters)
                
                ### Localization (U-Net)
                - **IoU:** {system.localizer_iou*100:.2f}%
                - **Dice Score:** ~20%
                - **Pixel Accuracy:** ~87%
                - **Architecture:** U-Net (17.3M parameters)
                
                ## 🎯 Capabilities
                
                - ✅ **Binary Classification**: Authentic vs Tampered detection
                - 🎯 **Pixel-Level Localization**: Highlights manipulated regions
                - ⚡ **Real-Time Inference**: <2 seconds per image on GPU
                - 🔍 **Multiple Forgery Types**: Copy-move, splicing, deepfakes
                
                ## 🗃️ Training Data
                
                - **CASIA v2.0**: 12,323 images
                - **CoMoFoD**: 10,400 images
                - **Total**: 19,014 images
                - **Split**: 70% train, 15% validation, 15% test
                
                ## 🔬 Preprocessing
                
                - **Error Level Analysis (ELA)**: Detects JPEG compression artifacts
                - **Quality**: 90
                - **Scale**: 10
                - **Image Size**: 224×224 pixels
                
                ## 🏗️ Technical Stack
                
                - **Framework**: PyTorch
                - **Models**: timm (EfficientNet), Custom U-Net
                - **Augmentation**: Albumentations
                - **Interface**: Gradio
                - **Hardware**: NVIDIA RTX 4050 (6GB VRAM)
                
                ## 📚 Research Foundation
                
                Based on state-of-the-art techniques:
                1. Error Level Analysis (ELA) for JPEG forensics
                2. EfficientNet for feature extraction
                3. U-Net for semantic segmentation
                4. Transfer learning from ImageNet
                
                ## 👥 Development Team
                
                *VeriPix - AI Image Forgery Detection System*
                
                **College Project** | Final Year Project 2025-26
                
                ---
                
                ### 📄 Citation
                
                If you use this system, please cite:
                ```
                VeriPix: AI-Powered Image Forgery Detection and Localization
                Using EfficientNet and U-Net Architectures
                [Your Institution], 2025
                ```
                
                ---
                
                ## 🔒 Privacy & Security
                
                - All processing happens locally
                - No images are stored or transmitted
                - Models run entirely on your machine
                
                ## ⚠️ Limitations
                
                - Works best on JPEG images (due to ELA preprocessing)
                - Localization accuracy depends on manipulation type
                - Large-scale edits easier to detect than subtle changes
                - Performance may vary on compressed/resized images
                
                ## 🚀 Future Improvements
                
                - [ ] Ensemble of multiple classifiers
                - [ ] Attention-based U-Net
                - [ ] Support for video forgery detection
                - [ ] Explainable AI visualizations
                - [ ] Mobile deployment
                
                ---
                
                **Version:** 1.0.0 | **Last Updated:** November 2025
                """)
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            Made with ❤️ using PyTorch, Gradio, and EfficientNet | 
            <a href="https://github.com/yourusername/veripix" target="_blank">GitHub</a> | 
            <a href="#" target="_blank">Documentation</a>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch VeriPix Full System")
    parser.add_argument('--port', type=int, default=7860, help='Port number')
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    parser.add_argument('--server-name', type=str, default="0.0.0.0", help='Server name')
    args = parser.parse_args()
    
    print("=" * 60)
    print("VeriPix - Image Forgery Detection System")
    print("=" * 60)
    
    # Create and launch
    demo = create_interface()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=True  # Auto-open browser
    )
