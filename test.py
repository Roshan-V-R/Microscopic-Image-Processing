import requests
import json
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

#base URL
BASE_URL = 'http://localhost:5000'

def display_image(img_path, title='Image'):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def test_upload_images():
    print("Testing image upload...")
    image_files = []
    for filename in os.listdir('sample_images'):
        if filename.endswith(('.jpg', '.png', '.tiff', '.jpeg')):
            image_path = os.path.join('sample_images', filename)
            image_files.append(('files', (filename, open(image_path, 'rb'), 'image/jpeg')))
    if not image_files:
        print("No sample images found in 'sample_images' directory")
        return None
    
    response = requests.post(f"{BASE_URL}/upload_images", files=image_files)
    for _, (_, file_obj, _) in image_files:
        file_obj.close()
    if response.status_code == 200:
        result = response.json()
        print(f"Successfully uploaded {len(result['filenames'])} images")
        return result['filenames']
    else:
        print(f"Failed to upload images: {response.text}")
        return None

def test_stitch_images(filenames):
    print("\nTesting image stitching...")
    if not filenames:
        print("No filenames provided for stitching")
        return None
    data = {"filenames": filenames}
    response = requests.post(f"{BASE_URL}/stitch_images", json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"Successfully stitched images: {result['result_path']}")
        return result['result_path']
    else:
        print(f"Failed to stitch images: {response.text}")
        return None

def test_roi_selection(image_path):
    print("\nTesting ROI selection...")
    if not image_path:
        print("No image path provided for ROI selection")
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image at {image_path}")
        return None
    height, width = img.shape[:2]
    roi_x = width // 4
    roi_y = height // 4
    roi_width = width // 2
    roi_height = height // 2
    
    data = {
        "image_path": image_path,
        "x": roi_x,
        "y": roi_y,
        "width": roi_width,
        "height": roi_height
    }

    response = requests.post(f"{BASE_URL}/roi_selection", json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"Successfully extracted ROI: {result['result_path']}")
        return result['result_path']
    else:
        print(f"Failed to extract ROI: {response.text}")
        return None

def test_zoom(image_path, zoom_factor=10):
    print(f"\nTesting {zoom_factor}X zoom...")
    
    if not image_path:
        print("No image path provided for zoom")
        return None
    
    data = {
        "image_path": image_path,
        "zoom_factor": zoom_factor
    }

    response = requests.post(f"{BASE_URL}/zoom", json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"Successfully applied {zoom_factor}X zoom: {result['result_path']}")
        return result['result_path']
    else:
        print(f"Failed to apply zoom: {response.text}")
        return None

def test_auto_focus(image_path):
    print("\nTesting auto-focus enhancement...")
    if not image_path:
        print("No image path provided for auto-focus")
        return None
    
    data = {
        "image_path": image_path
    }

    response = requests.post(f"{BASE_URL}/auto_focus", json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"Successfully applied auto-focus: {result['result_path']}")
        return result['result_path']
    else:
        print(f"Failed to apply auto-focus: {response.text}")
        return None

def download_image(image_path):
    if not image_path:
        return None
    
    filename = os.path.basename(image_path)
    response = requests.get(f"{BASE_URL}/get_image/{filename}")
    if response.status_code == 200:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Image saved to {output_path}")
        return output_path
    else:
        print(f"Failed to download image: {response.text}")
        return None

def run_complete_pipeline():
    print("=== Starting Microscope Image Processing Pipeline ===")
    
    # Step 1: Upload images
    filenames = test_upload_images()
    if not filenames:
        return
    
    # Step 2: Stitch images
    stitched_image_path = test_stitch_images(filenames)
    if not stitched_image_path:
        return
    
    # Download and display stitched image
    downloaded_stitched = download_image(stitched_image_path)
    if downloaded_stitched:
        display_image(downloaded_stitched, "Stitched Image")
    
    # Step 3: Extract ROI
    roi_path = test_roi_selection(stitched_image_path)
    if not roi_path:
        return
    
    # Download and display ROI
    downloaded_roi = download_image(roi_path)
    if downloaded_roi:
        display_image(downloaded_roi, "Selected ROI")
    
    # Step 4: Apply 10X zoom
    zoomed_10x_path = test_zoom(roi_path, 10)
    if not zoomed_10x_path:
        return
    
    # Download and display 10X zoomed image
    downloaded_zoom_10x = download_image(zoomed_10x_path)
    if downloaded_zoom_10x:
        display_image(downloaded_zoom_10x, "10X Zoom")
    
    # Step 5: Apply 20X zoom on a different ROI
    zoomed_20x_path = test_zoom(roi_path, 20)
    if not zoomed_20x_path:
        return
    
    # Download and display 20X zoomed image
    downloaded_zoom_20x = download_image(zoomed_20x_path)
    if downloaded_zoom_20x:
        display_image(downloaded_zoom_20x, "20X Zoom")
    
    # Step 6: Apply auto-focus enhancement
    enhanced_path = test_auto_focus(zoomed_20x_path)
    if not enhanced_path:
        return
    
    # Download and display enhanced image
    downloaded_enhanced = download_image(enhanced_path)
    if downloaded_enhanced:
        display_image(downloaded_enhanced, "Enhanced Image (Auto-Focus)")
    
    print("=== Microscope Image Processing Pipeline Completed Successfully ===")

if __name__ == "__main__":
    os.makedirs('sample_images', exist_ok=True)
    if not [f for f in os.listdir('sample_images') if f.endswith(('.jpg', '.png', '.tiff', '.jpeg'))]:
        print("Please place sample microscope images in the 'sample_images' directory before running this script.")
    else:
        run_complete_pipeline()