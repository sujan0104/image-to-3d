import torch
import cv2
import numpy as np

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # or "MiDaS_small" for speed
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform  # for DPT_Large and DPT_Hybrid

# Read image
img_path = "data/sample.jpg"  # change if needed
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess
input_batch = transform(img).to(device)

# Sanity check shape
print("Input shape:", input_batch.shape)  # should be [1, 3, H, W]

# Inference
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()

depth_map = prediction.cpu().numpy()

# Normalize for saving
depth_min = depth_map.min()
depth_max = depth_map.max()
depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)
depth_map_normalized = (depth_map_normalized * 255).astype(np.uint8)

# Save output
cv2.imwrite("output_depth.png", depth_map_normalized)
print("Depth map saved to output_depth.png")
