# pneumonia_eval.py
import os, math, random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.lines as mlines
import pandas as pd


# -----------------------------
# Paths and device
# -----------------------------
DATA_DIR   = "/home/jason.tieh/pneumonia_project"
OUTPUT_DIR = "/home/jason.tieh/pneumonia_project_outputs"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDS_DIR   = os.path.join(RESULTS_DIR, "predictions")
GRAD_DIR    = os.path.join(RESULTS_DIR, "gradcam")
GRADPP_DIR  = os.path.join(RESULTS_DIR, "gradcam++")
for folder in [RESULTS_DIR, PREDS_DIR, GRAD_DIR, GRADPP_DIR]:
    os.makedirs(folder, exist_ok=True)

# -----------------------------
# Load test dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
CLASS_NAMES = test_ds.classes
print(f"Test dataset loaded: {len(test_ds)} images, classes: {CLASS_NAMES}")

# -----------------------------
# Load model
# -----------------------------
model = models.densenet121(pretrained=True)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_pneumonia_model.pth"), map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

# -----------------------------
# Denormalization helper
# -----------------------------
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])
def denorm(tensor):
    img = tensor.permute(1,2,0).cpu().numpy()
    img = (img*STD)+MEAN
    return np.clip(img,0,1).astype(np.float32)

# -----------------------------
# Evaluation
# -----------------------------
y_true, y_pred, y_prob = [], [], []
print("Running predictions on test set...")
with torch.no_grad():
    for idx, (img, label) in enumerate(test_ds):
        input_tensor = img.unsqueeze(0).to(DEVICE)
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()
        y_true.append(label)
        y_pred.append(pred)
        y_prob.append(prob[0,1].item())
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(test_ds)} images")

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()
print("Saved confusion_matrix.png")

# -----------------------------
# ROC curve
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

roc_df = pd.DataFrame({
    "Threshold": thresholds,
    "FPR": fpr,
    "TPR": tpr
})

roc_df.to_csv(os.path.join(RESULTS_DIR, "roc_values.csv"), index=False)
print("Saved roc_values.csv")

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={roc_auc:.4f}')
plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close()
print("Saved roc_curve.png")

# -----------------------------
# Classification report
# -----------------------------
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
with open(os.path.join(RESULTS_DIR, "test_report.txt"), "w") as f:
    f.write(f"Pneumonia Detection Test Report\nGenerated: {datetime.now()}\nROC AUC: {roc_auc:.4f}\n\n")
    f.write(report)
print("Saved test_report.txt")
print(report)

# -----------------------------
# Grad-CAM setup
# -----------------------------
target_layers = [model.features[-1]]
cam_gc = GradCAM(model=model, target_layers=target_layers)
cam_gcpp = GradCAMPlusPlus(model=model, target_layers=target_layers)
print("Grad-CAM objects initialized.")

# -----------------------------
# Build sample buckets
# -----------------------------
bucket_correct_normal, bucket_correct_pneumonia, bucket_wrong = [], [], []
for idx in range(len(test_ds)):
    img, label = test_ds[idx]
    input_tensor = img.unsqueeze(0).to(DEVICE)
    output = model(input_tensor)
    pred = output.argmax(1).item()
    entry = (idx,label,pred)
    if label != pred:
        bucket_wrong.append(entry)
    elif label == 0:
        bucket_correct_normal.append(entry)
    else:
        bucket_correct_pneumonia.append(entry)

print(f"Buckets: NORMAL={len(bucket_correct_normal)}, PNEUMONIA={len(bucket_correct_pneumonia)}, WRONG={len(bucket_wrong)}")

# -----------------------------
# Choose samples for galleries
# -----------------------------
random.seed(42)
pred_sample_list = (random.sample(bucket_correct_normal, min(8,len(bucket_correct_normal))) +
                    random.sample(bucket_correct_pneumonia, min(8,len(bucket_correct_pneumonia))) +
                    random.sample(bucket_wrong, min(4,len(bucket_wrong))))
cam_sample_list  = (random.sample(bucket_correct_normal, min(5,len(bucket_correct_normal))) +
                    random.sample(bucket_correct_pneumonia, min(5,len(bucket_correct_pneumonia))) +
                    random.sample(bucket_wrong, min(2,len(bucket_wrong))))

# -----------------------------
# Generate per-sample images
# -----------------------------
render_store = {}
print("Generating individual prediction and Grad-CAM images...")
for i, (idx, label, pred) in enumerate(pred_sample_list + cam_sample_list):
    if idx in render_store:
        continue
    img_tensor, _ = test_ds[idx]
    input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    img_show = denorm(img_tensor)
    correct_mark = "✓" if label == pred else "✗"
    title_short = f"{correct_mark} {CLASS_NAMES[label]}→{CLASS_NAMES[pred]}"
    # Grad-CAM
    gc_map = cam_gc(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred)])[0]
    vis_gc = show_cam_on_image(img_show, gc_map, use_rgb=True)
    gcpp_map = cam_gcpp(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred)])[0]
    vis_gcpp = show_cam_on_image(img_show, gcpp_map, use_rgb=True)
    render_store[idx] = {"orig": img_show, "gc": vis_gc, "gcpp": vis_gcpp, "is_correct": label==pred, "title_short": title_short}
    # Save images
    plt.imsave(os.path.join(PREDS_DIR, f"prediction_{i:02d}.png"), img_show)
    plt.imsave(os.path.join(GRAD_DIR, f"gradcam_{i:02d}.png"), vis_gc)
    plt.imsave(os.path.join(GRADPP_DIR, f"gradcampp_{i:02d}.png"), vis_gcpp)
    if i % 10 == 0:
        print(f"Saved {i+1} images...")

# -----------------------------
# Create galleries
# -----------------------------
print("Creating galleries...")

# Prediction gallery
G_ROWS, G_COLS = 5,4
fig, axes = plt.subplots(G_ROWS, G_COLS, figsize=(G_COLS*4, G_ROWS*4))
for idx, ax in enumerate(axes.flat):
    if idx < len(pred_sample_list):
        gidx = pred_sample_list[idx][0]
        d = render_store[gidx]
        ax.imshow(d['orig'])
        ax.set_title(d['title_short'], fontsize=8, color='green' if d['is_correct'] else 'red')
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'gallery_predictions.png'), dpi=150)
plt.close()
print("Saved gallery_predictions.png")

# Grad-CAM gallery
fig, axes = plt.subplots(6,6, figsize=(6*4,6*4))
for row in range(6):
    for side in range(2):
        sample_idx = row*2+side
        col_base = side*3
        if sample_idx < len(cam_sample_list):
            gidx = cam_sample_list[sample_idx][0]
            d = render_store[gidx]
            color = 'green' if d['is_correct'] else 'red'
            axes[row][col_base].imshow(d['orig'])
            axes[row][col_base+1].imshow(d['gc'])
            axes[row][col_base+2].imshow(d['gcpp'])
            axes[row][col_base].set_title(d['title_short'], fontsize=7.5, color=color)
            axes[row][col_base+1].set_title('Grad-CAM', fontsize=7, color='dimgray')
            axes[row][col_base+2].set_title('Grad-CAM++', fontsize=7, color='dimgray')
        for c in range(col_base, col_base+3):
            axes[row][c].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'gallery_gradcam.png'), dpi=150)
plt.close()
print("Saved gallery_gradcam.png")

print("Evaluation complete! All results saved in:", RESULTS_DIR)