import cv2
import sys
from pathlib import Path

def load_image(image_path):
    """Loads an image using OpenCV."""
    if not Path(image_path).exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)
    
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Failed to load image.")
        sys.exit(1)
    
    print("[INFO] Image loaded successfully.")
    return image

def main():
    # Example usage: change this path to your image file
    image_path = "data/sample.jpg"
    
    img = load_image(image_path)
    
    # Display the image
    cv2.imshow("Loaded Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
