import cv2
from tkinter import Tk, filedialog
import sys
import os
import shutil

# Paths
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "../data")

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# --- Step 1: Select Image ---
Tk().withdraw()  # Hide the Tkinter root window

file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if not file_path:
    print("[ERROR] No image selected.")
    sys.exit()

# --- Step 2: Clean data folder ---
for file in os.listdir(DATA_FOLDER):
    try:
        os.remove(os.path.join(DATA_FOLDER, file))
    except Exception as e:
        print(f"[WARN] Could not delete {file}: {e}")

# --- Step 3: Copy Image into data folder ---
file_name = os.path.basename(file_path)
dest_path = os.path.join(DATA_FOLDER, file_name)

try:
    shutil.copy(file_path, dest_path)
    print(f"[INFO] Image copied to data folder: {dest_path}")
except Exception as e:
    print(f"[ERROR] Failed to copy image: {e}")
    sys.exit()

# --- Step 4: Load Image from data folder ---
image = cv2.imread(dest_path)
if image is None:
    print(f"[ERROR] Unable to load image: {dest_path}")
    sys.exit()

print(f"[INFO] Image loaded successfully: {file_name}")

# --- Step 5: Continue with your 3D processing pipeline ---
# Replace this with your actual 3D processing code
cv2.imshow("Selected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
