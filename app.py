from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import uuid
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs(PROCESSED_FOLDER,exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to save uploaded files
def save_uploaded_files(files):
    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            filenames.append(filepath)
    return filenames

# Image stitching function using OpenCV
def stitch_images(image_paths):
    logger.info(f"Starting image stitching for {len(image_paths)} images")
    
    # Reading all images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            logger.error(f"Failed to read image:{path}")
            continue
        images.append(img)
    
    if len(images)<2:
        raise ValueError("Need at least two images for stitching")
    
    # Initializing ORB
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Finding keypoints and descriptors
    keypoints = []
    descriptors = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
    
    # Create matcher and match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Define the base image (middle one to minimize distortion)
    base_index = len(images)//2
    base_img = images[base_index]
    
    # Stitch each image to the base image
    stitched_img = base_img.copy()
    
    for i in range(len(images)):
        if i == base_index:
            continue
            
        matches = matcher.match(descriptors[base_index],descriptors[i])
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.8)]
        
        if len(good_matches)<4:
            logger.warning(f"Not enough matches found between base image and image {i}")
            continue
        
        src_pts = np.float32([keypoints[base_index][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Finding homography
        H,_ = cv2.findHomography(dst_pts,src_pts,cv2.RANSAC,5.0)
        
        # Warping the image
        h,w = base_img.shape[:2]
        warped_img = cv2.warpPerspective(images[i],H,(w*2, h*2))
        mask = np.zeros((h*2,w*2,3), dtype=np.uint8)
        mask[0:h, 0:w] = (255,255,255)
        
        # Blending the images
        warped_mask = cv2.warpPerspective(np.ones_like(images[i]) * 255, H, (w*2, h*2))
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_BGR2GRAY)
        _,warped_mask = cv2.threshold(warped_mask, 0, 255, cv2.THRESH_BINARY)
        
        expanded_stitched = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        expanded_stitched[0:h, 0:w] = stitched_img
       
        original_mask = np.zeros((h*2, w*2), dtype=np.uint8)
        original_mask[0:h, 0:w] = 255
        
        overlap_mask = cv2.bitwise_and(original_mask, warped_mask)
       
        non_overlap_original = cv2.bitwise_and(original_mask, cv2.bitwise_not(overlap_mask))
        non_overlap_warped = cv2.bitwise_and(warped_mask, cv2.bitwise_not(overlap_mask))
        
        result = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        for c in range(3):
            result[:, :, c] = result[:, :, c] + expanded_stitched[:, :, c] * (non_overlap_original / 255)
            result[:, :, c] = result[:, :, c] + warped_img[:, :, c] * (non_overlap_warped / 255)
        
        overlap_area = overlap_mask / 255
        for c in range(3):
            result[:, :, c] = result[:, :, c] + (expanded_stitched[:, :, c] + warped_img[:, :, c]) / 2 * overlap_area
        
        result = result.astype(np.uint8)
        
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))
            result = result[y:y+h, x:x+w]
        
        stitched_img = result
    

    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        stitched_img = stitched_img[y:y+h, x:x+w]
    
    logger.info("Image stitching completed successfully")
    return stitched_img

# Extract Region of Interest
def extract_roi(image_path, x, y, width, height):
    logger.info(f"Extracting ROI at ({x}, {y}) with dimensions {width}x{height}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Validate ROI parameters
    if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
        raise ValueError("ROI coordinates are outside image boundaries")
    
    # Extract ROI
    roi = img[y:y+height, x:x+width]
    
    return roi

# Apply digital zoom
def apply_zoom(image, zoom_factor):
    logger.info(f"Applying {zoom_factor}X zoom")
    
    if zoom_factor not in [10, 20]:
        raise ValueError("Zoom factor must be either 10X or 20X")
    
    height, width = image.shape[:2]
    new_height = int(height * zoom_factor / 10)  
    new_width = int(width * zoom_factor / 10)
    zoomed_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return zoomed_img

# Apply auto-focus (sharpening)
def apply_auto_focus(image):
    logger.info("Applying auto-focus enhancement")
    
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
   
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    logger.info(f"Image sharpness value: {sharpness}")
    
    if sharpness < 100:
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        enhanced = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        logger.info("Applied sharpening filter due to low sharpness")
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        logger.info("Applied contrast enhancement to already sharp image")
    
    return enhanced

# API Endpoints

@app.route('/upload_images', methods=['POST'])
def upload_images():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    filenames = save_uploaded_files(files)
    
    if not filenames:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    return jsonify({
        'status': 'success',
        'message': f'Successfully uploaded {len(filenames)} images',
        'filenames': filenames
    })

@app.route('/stitch_images', methods=['POST'])
def stitch_images_endpoint():
    data = request.get_json()
    if not data or 'filenames' not in data:
        return jsonify({'error': 'No filenames provided'}), 400
    
    image_paths = data['filenames']
    
    try:
        stitched_img = stitch_images(image_paths)
        result_filename = f"stitched_{uuid.uuid4()}.jpg"
        result_path = os.path.join(app.config['PROCESSED_FOLDER'], result_filename)
        cv2.imwrite(result_path, stitched_img)
        
        return jsonify({
            'status': 'success',
            'message': 'Images stitched successfully',
            'result_path': result_path
        })
    
    except Exception as e:
        logger.error(f"Error in image stitching: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/roi_selection', methods=['POST'])
def roi_selection_endpoint():
    data = request.get_json()
    required_fields = ['image_path', 'x', 'y', 'width', 'height']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        roi = extract_roi(
            data['image_path'],
            int(data['x']),
            int(data['y']),
            int(data['width']),
            int(data['height'])
        )
        
      
        result_filename = f"roi_{uuid.uuid4()}.jpg"
        result_path = os.path.join(app.config['PROCESSED_FOLDER'], result_filename)
        cv2.imwrite(result_path, roi)
        
        return jsonify({
            'status': 'success',
            'message': 'ROI extracted successfully',
            'result_path': result_path
        })
    
    except Exception as e:
        logger.error(f"Error in ROI extraction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/zoom', methods=['POST'])
def zoom_endpoint():
    data = request.get_json()
    if not data or 'image_path' not in data or 'zoom_factor' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        img = cv2.imread(data['image_path'])
        if img is None:
            return jsonify({'error': f"Failed to read image: {data['image_path']}"}), 400
        
        zoom_factor = int(data['zoom_factor'])
        zoomed_img = apply_zoom(img, zoom_factor)
        result_filename = f"zoomed_{zoom_factor}x_{uuid.uuid4()}.jpg"
        result_path = os.path.join(app.config['PROCESSED_FOLDER'], result_filename)
        cv2.imwrite(result_path, zoomed_img)
        
        return jsonify({
            'status': 'success',
            'message': f'Applied {zoom_factor}X zoom successfully',
            'result_path': result_path
        })
    
    except Exception as e:
        logger.error(f"Error in zoom application: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/auto_focus', methods=['POST'])
def auto_focus_endpoint():
    data = request.get_json()
    
    if not data or 'image_path' not in data:
        return jsonify({'error': 'Missing image_path'}), 400
    
    try:
        img = cv2.imread(data['image_path'])
        if img is None:
            return jsonify({'error': f"Failed to read image: {data['image_path']}"}), 400
        
        enhanced_img = apply_auto_focus(img)
        result_filename = f"enhanced_{uuid.uuid4()}.jpg"
        result_path = os.path.join(app.config['PROCESSED_FOLDER'], result_filename)
        cv2.imwrite(result_path, enhanced_img)
        
        return jsonify({
            'status': 'success',
            'message': 'Image enhanced successfully',
            'result_path': result_path
        })
    
    except Exception as e:
        logger.error(f"Error in auto-focus enhancement: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_image/<filename>', methods=['GET'])
def get_image(filename):
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(filepath, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)