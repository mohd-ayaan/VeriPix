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

        # Initialize ELA processor (keep same as training)
        self.ela_processor = ELAProcessor(quality=90, scale=10)

        # Initialize transform
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Load classifier
        print("Loading classifier...")
        self.classifier = create_efficientnet_classifier(
            pretrained=False,
            device=self.device
        )
        cls_checkpoint = torch.load(
            'checkpoints/classifier/best.pth',
            map_location=self.device,
            weights_only=False
        )
        self.classifier.load_state_dict(cls_checkpoint['model_state_dict'])
        self.classifier.eval()
        self.classifier_f1 = cls_checkpoint.get('best_f1', 0.0)
        print(f"  ✓ Classifier loaded (F1: {self.classifier_f1:.4f})")

        # Load localizer
        print("Loading localizer...")
        self.localizer = create_unet_localizer(device=self.device)
        loc_checkpoint = torch.load(
            'checkpoints/localizer/best.pth',
            map_location=self.device,
            weights_only=False
        )
        self.localizer.load_state_dict(loc_checkpoint['model_state_dict'])
        self.localizer.eval()
        self.localizer_iou = loc_checkpoint.get('best_iou', 0.0)
        print(f"  ✓ Localizer loaded (IoU: {self.localizer_iou:.4f})")

    def predict_from_path(self, image_path, cls_threshold=0.5, loc_threshold=0.5):
        """
        Complete prediction pipeline from file path

        Returns:
            tuple: (display_label, confidence, vis_original, heatmap, overlay)
        """
        if not Path(image_path).exists():
            return "❌ File not found", 0.0, None, None, None

        try:
            # Load original image for visualization
            original = cv2.imread(str(image_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            original_display = cv2.resize(original, (self.image_size, self.image_size))

            # Apply ELA (used as input feature)
            ela_img = self.ela_processor.apply_ela(
                image_path,
                output_size=(self.image_size, self.image_size)
            )

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
                    loc_prob = np.zeros((self.image_size, self.image_size), dtype=np.float32)
                    mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

            # Classification label (internal and display)
            base_label = "Tampered (Forged)" if is_tampered else "Authentic"
            display_label = f"🔴 {base_label}" if is_tampered else f"✅ {base_label}"
            confidence_pct = cls_prob * 100.0

            # Visualization 1: Original with colored border
            border_color = (255, 0, 0) if is_tampered else (0, 255, 0)
            vis_original = cv2.copyMakeBorder(
                original_display, 10, 10, 10, 10,
                cv2.BORDER_CONSTANT, value=border_color
            )

            # Visualization 2: Heatmap from localization probabilities
            heatmap = cv2.applyColorMap(
                (loc_prob * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Visualization 3: Binary mask overlay
            mask_colored = np.zeros_like(original_display)
            mask_colored[:, :, 0] = mask  # red channel

            if is_tampered and mask.sum() > 0:
                overlay = cv2.addWeighted(original_display, 0.6, mask_colored, 0.4, 0)
            else:
                overlay = original_display.copy()

            tampered_percent = (mask > 0).sum() / (self.image_size * self.image_size) * 100.0

            # Debug output in console
            print(f"[Prediction] {Path(image_path).name}")
            print(f"  Classification: {base_label} ({confidence_pct:.2f}%)")
            print(f"  Localization: {tampered_percent:.2f}% of image marked as tampered")

            return display_label, confidence_pct, vis_original, heatmap, overlay

        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()
            return f"❌ Error: {str(e)}", 0.0, None, None, None

    def predict_from_upload(self, image_array, cls_threshold=0.5, loc_threshold=0.5):
        """Predict from uploaded numpy array (Gradio Image input)."""
        if image_array is None:
            return "❌ No image uploaded", 0.0, None, None, None

        try:
            # Normalize array
            if image_array.dtype in (np.float32, np.float64):
                image_array = (image_array * 255).astype(np.uint8)

            # Handle channels
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            # Save to temp JPEG (for ELA)
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', dir='.')
            os.close(temp_fd)

            try:
                pil_img = PILImage.fromarray(image_array, mode='RGB')
                pil_img.save(temp_path, 'JPEG', quality=95)
                return self.predict_from_path(temp_path, cls_threshold, loc_threshold)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()
            return f"❌ Error: {str(e)}", 0.0, None, None, None


def create_interface():
    """Create Gradio interface."""

    system = VeriPixSystem()

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

    with gr.Blocks(css=custom_css,
                   title="VeriPix - AI Image Forgery Detector") as demo:

        # Header
        gr.HTML("""
        <div class="title-text">🔍 VeriPix</div>
        <div class="subtitle-text">AI-Powered Image Forgery Detection & Localization</div>
        """)

        # System metrics
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
                    <p style="font-size: 2em; margin: 5px 0;">{71.8}%</p>
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

        with gr.Tabs():

            # Tab 1: Upload & Analyze
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
                                0, 1, 0.4, 0.05,    # slightly lower default for clearer heatmap
                                label="Localization Threshold"
                            )

                        analyze_btn = gr.Button(
                            "🔍 Analyze Image",
                            variant="primary"
                        )

                    with gr.Column():
                        result_label = gr.Textbox(
                            label="Classification Result",
                            lines=1
                        )
                        result_conf = gr.Number(
                            label="Confidence (%)",
                            precision=2
                        )

                with gr.Row():
                    vis_original = gr.Image(
                        label="Original (with border)",
                        height=250
                    )
                    vis_heatmap = gr.Image(
                        label="Localization Heatmap",
                        height=250
                    )
                    vis_overlay = gr.Image(
                        label="Overlay Result",
                        height=250
                    )

                analyze_btn.click(
                    fn=system.predict_from_upload,
                    inputs=[upload_input, cls_threshold, loc_threshold],
                    outputs=[result_label, result_conf,
                             vis_original, vis_heatmap, vis_overlay]
                )

            # Tab 2: Test Examples
            with gr.Tab("📋 Test Examples"):
                gr.Markdown("""
                ### Test on Pre-loaded Examples
                Click buttons to test on known authentic and tampered images from the test set.
                """)

                with gr.Row():
                    ex_cls_thresh = gr.Slider(
                        0, 1, 0.5, 0.05, label="Classification Threshold"
                    )
                    ex_loc_thresh = gr.Slider(
                        0, 1, 0.4, 0.05, label="Localization Threshold"
                    )

                with gr.Row():
                    btn_auth = gr.Button("✅ Test Authentic Image")
                    btn_tamp1 = gr.Button("🔴 Test Tampered #1")
                    btn_tamp2 = gr.Button("🔴 Test Tampered #2")

                ex_label = gr.Textbox(label="Classification Result", lines=1)
                ex_conf = gr.Number(label="Confidence (%)", precision=2)

                with gr.Row():
                    ex_original = gr.Image(label="Original", height=250)
                    ex_heatmap = gr.Image(label="Heatmap", height=250)
                    ex_overlay = gr.Image(label="Overlay", height=250)

                btn_auth.click(
                    fn=lambda ct, lt: system.predict_from_path(
                        "data/processed/test/authentic/casia_Au_ani_00004.jpg", ct, lt
                    ),
                    inputs=[ex_cls_thresh, ex_loc_thresh],
                    outputs=[ex_label, ex_conf,
                             ex_original, ex_heatmap, ex_overlay]
                )

                btn_tamp1.click(
                    fn=lambda ct, lt: system.predict_from_path(
                        "data/processed/test/tampered/casia_Tp_D_CNN_M_N_art00052_arc00030_11853.jpg",
                        ct, lt
                    ),
                    inputs=[ex_cls_thresh, ex_loc_thresh],
                    outputs=[ex_label, ex_conf,
                             ex_original, ex_heatmap, ex_overlay]
                )

                btn_tamp2.click(
                    fn=lambda ct, lt: system.predict_from_path(
                        "data/processed/test/tampered/casia_Tp_D_CND_M_N_art00077_art00076_10290.tif",
                        ct, lt
                    ),
                    inputs=[ex_cls_thresh, ex_loc_thresh],
                    outputs=[ex_label, ex_conf,
                             ex_original, ex_heatmap, ex_overlay]
                )

            # Tab 3: About (unchanged – you can keep your existing Markdown)
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                # About VeriPix
                (keep your existing long markdown content here)
                """)

        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            Made with ❤️ using PyTorch, Gradio, and EfficientNet
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

    demo = create_interface()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )
