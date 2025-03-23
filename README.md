# Microscope Image Processing API

A Flask-based REST API for processing microscope images with OpenCV.

## Features

- **Image Stitching**: Combines multiple overlapping microscope images into a single high-resolution panorama
- **ROI Selection**: Extracts specific regions of interest from images using coordinates
- **Digital Zoom**: Magnifies selected regions at 10X and 20X levels using bicubic interpolation
- **Auto-Focus Simulation**: Enhances image clarity using contrast-based sharpening techniques

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/microscope-image-api.git
   cd microscope-image-api
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

   The API will be available at `http://localhost:5000`

## API Endpoints

### 1. Upload Images

Upload microscope images for processing.

- **URL**: `/upload_images`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Request Body**:
  - `files`: Multiple image files (supported formats: PNG, JPG, JPEG, TIF, TIFF)
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Successfully uploaded 3 images",
    "filenames": [
      "uploads/12345_image1.jpg",
      "uploads/67890_image2.jpg",
      "uploads/24680_image3.jpg"
    ]
  }
  ```

### 2. Stitch Images

Stitch multiple overlapping images into a single panoramic image.

- **URL**: `/stitch_images`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "filenames": [
      "uploads/12345_image1.jpg",
      "uploads/67890_image2.jpg",
      "uploads/24680_image3.jpg"
    ]
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Images stitched successfully",
    "result_path": "processed/stitched_abcdef.jpg"
  }
  ```

### 3. ROI Selection

Extract a region of interest from an image.

- **URL**: `/roi_selection`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "image_path": "processed/stitched_abcdef.jpg",
    "x": 100,
    "y": 150,
    "width": 200,
    "height": 200
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "ROI extracted successfully",
    "result_path": "processed/roi_123456.jpg"
  }
  ```

### 4. Zoom

Apply digital zoom to an image.

- **URL**: `/zoom`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "image_path": "processed/roi_123456.jpg",
    "zoom_factor": 10
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Applied 10X zoom successfully",
    "result_path": "processed/zoomed_10x_789012.jpg"
  }
  ```

### 5. Auto-Focus

Enhance image sharpness and clarity.

- **URL**: `/auto_focus`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "image_path": "processed/zoomed_10x_789012.jpg"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Image enhanced successfully",
    "result_path": "processed/enhanced_345678.jpg"
  }
  ```

### 6. Get Image

Retrieve a processed image.

- **URL**: `/get_image/<filename>`
- **Method**: `GET`
- **Response**: The requested image file.

## Technical Implementation Details

### Image Stitching

The image stitching process follows these steps:

1. **Feature Detection**: Uses ORB algorithm to detect keypoints and compute descriptors for each image
2. **Feature Matching**: Employs Brute Force Matcher to find corresponding points between images
3. **Homography Calculation**: Computes the transformation matrix to align images
4. **Image Warping**: Applies the transformation to align all images to a common frame
5. **Image Blending**: Smoothly combines the aligned images to create a seamless panorama

### ROI Selection

The ROI selection process:
1. Validates that the requested region falls within the image boundaries
2. Extracts the specified rectangle using OpenCV's array slicing operations

### Digital Zoom

The zoom implementation:
1. Uses bicubic interpolation for high-quality resizing
2. Normalizes zoom factors based on 10X as the reference point
3. Produces smoother images by preserving edge details during magnification

### Auto-Focus Simulation

The auto-focus algorithm:
1. Calculates the Laplacian variance as a measure of image sharpness
2. For low-sharpness images, applies unsharp masking for enhancement
3. For already-sharp images, applies mild CLAHE contrast enhancement

## Performance Optimizations

1. **Memory Efficiency**:
   - Images are read only when needed and not stored in memory
   - Process large images in chunks when appropriate

2. **Processing Optimizations**:
   - Uses ORB instead of SIFT/SURF for faster feature detection
   - Implements automatic cropping of black borders after stitching
   - Applies different enhancement techniques based on image properties

3. **API Design**:
   - Stateless design allows for horizontal scaling
   - Unique file naming prevents collisions in concurrent operations
   - Proper error handling with informative messages

## Challenges and Solutions

### Challenge 1: Handling Large Microscope Images

**Problem**: Microscope images can be very large (10-100MB), causing memory issues during processing.

**Solution**: Implemented streaming uploads with chunked processing and added maximum size constraints with proper error handling.

### Challenge 2: Accurate Image Stitching

**Problem**: Microscope images often have subtle overlaps that are difficult to detect.

**Solution**: Enhanced feature detection by using ORB with higher feature counts and implemented better matching with RANSAC to handle outliers.

### Challenge 3: Maintaining Image Quality During Zoom

**Problem**: Digital zoom can introduce artifacts and blur.

**Solution**: Used bicubic interpolation instead of bilinear for better quality and applied subtle sharpening after zooming.

## Sample Usage Flow

A typical workflow using this API:

1. Upload multiple microscope image sections:
   ```
   POST /upload_images with image files
   ```

2. Stitch the images into a complete view:
   ```
   POST /stitch_images with the uploaded filenames
   ```

3. Select a region of interest:
   ```
   POST /roi_selection with coordinates
   ```

4. Apply zoom for detailed examination:
   ```
   POST /zoom with the ROI image path
   ```

5. Enhance the clarity with auto-focus:
   ```
   POST /auto_focus with the zoomed image path
   ```

6. Retrieve the final processed image:
   ```
   GET /get_image/enhanced_filename.jpg
   ```

## Future Improvements

1. Add image annotation features for marking structures of interest
2. Implement batch processing for large sets of microscope images
3. Add image segmentation for automatic cell/structure detection
4. Create a simple web frontend for easier interaction with the API
5. Add authentication and user management for access control
