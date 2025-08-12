import os
import shutil
import torch
import cv2
import numpy as np
import open3d as o3d
from tkinter import Tk, filedialog

# Setup paths
BASE_DIR = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_DIR, "../data")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "../outputs")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Step 1: Select image file ---
Tk().withdraw()  # Hide Tkinter root window
file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)
if not file_path:
    print("No image selected. Exiting.")
    exit()

# --- Step 2: Clean data folder and copy selected image ---
for f in os.listdir(DATA_FOLDER):
    os.remove(os.path.join(DATA_FOLDER, f))
dest_path = os.path.join(DATA_FOLDER, os.path.basename(file_path))
shutil.copy(file_path, dest_path)
print(f"Copied image to data folder: {dest_path}")

# --- Step 3: Load MiDaS model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# --- Step 4: Read image ---
img = cv2.imread(dest_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {dest_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Step 5: Preprocess and predict depth ---
input_batch = transform(img_rgb).to(device)
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()
depth_map = prediction.cpu().numpy()

# --- Step 6: Normalize depth ---
depth_min, depth_max = depth_map.min(), depth_map.max()
depth_norm = (depth_map - depth_min) / (depth_max - depth_min)

# --- Step 7: Save grayscale and colorized depth maps ---
depth_uint8 = (depth_norm * 255).astype(np.uint8)
grayscale_path = os.path.join(OUTPUT_FOLDER, "depth_grayscale.png")
cv2.imwrite(grayscale_path, depth_uint8)
depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
color_path = os.path.join(OUTPUT_FOLDER, "depth_colorized.png")
cv2.imwrite(color_path, depth_color)
print(f"Saved depth maps:\n - {grayscale_path}\n - {color_path}")

# --- Step 8: Create 3D point cloud and save ---
color_o3d = o3d.geometry.Image(img_rgb)
depth_o3d = o3d.geometry.Image((depth_norm * 1000).astype(np.uint16))
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d,
    depth_scale=1000.0,
    depth_trunc=3.0,
    convert_rgb_to_intensity=False
)
h, w = depth_norm.shape
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    w, h,
    fx=500, fy=500,
    cx=w // 2, cy=h // 2
)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])
ply_path = os.path.join(OUTPUT_FOLDER, "point_cloud.ply")
o3d.io.write_point_cloud(ply_path, pcd)
print(f"3D point cloud saved to {ply_path}")
