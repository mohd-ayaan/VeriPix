import torch

# Load classifier checkpoint (with weights_only=False for PyTorch 2.6+)
cls_checkpoint = torch.load('checkpoints/classifier/best.pth', weights_only=False)
print("=" * 60)
print("CLASSIFIER RESULTS")
print("=" * 60)
print(f"Best F1-Score: {cls_checkpoint.get('best_f1', 'N/A'):.4f}")
print(f"Trained for {cls_checkpoint.get('epoch', 'N/A') + 1} epochs")

# Load localizer checkpoint
try:
    loc_checkpoint = torch.load('checkpoints/localizer/best.pth', weights_only=False)
    print("\n" + "=" * 60)
    print("LOCALIZER RESULTS")
    print("=" * 60)
    print(f"Best IoU: {loc_checkpoint.get('best_iou', 'N/A'):.4f}")
    print(f"Trained for {loc_checkpoint.get('epoch', 'N/A') + 1} epochs")
except Exception as e:
    print("\n" + "=" * 60)
    print("LOCALIZER RESULTS")
    print("=" * 60)
    print(f"Error loading localizer: {e}")
    print("Note: Localizer checkpoint may be corrupted or incomplete")
