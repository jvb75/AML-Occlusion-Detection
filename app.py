from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'models/yolov8m_v1.pt'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully. Classes: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_heads(image_path, conf_threshold=0.1):
    if not model:
        return [], "Model not loaded"
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return [], "Could not read image"
        
        results = model(img, conf=conf_threshold)
        boxes = [[int(x) for x in box.xyxy[0].tolist()] for result in results for box in result.boxes]
        return boxes, "Success"
    except Exception as e:
        return [], f"Detection error: {str(e)}"

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection) if (area1 + area2 - intersection) > 0 else 0

def find_occlusions(boxes, iou_threshold=0.2):
    occluded_indices = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                occluded_indices.add(i)
                occluded_indices.add(j)
    return [boxes[i] for i in occluded_indices]

# Handle 413 Request Entity Too Large
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File(s) exceed the 10MB limit'}), 413

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    try:
        conf_threshold = float(request.form.get('conf_threshold', 0.1))
        iou_threshold = float(request.form.get('iou_threshold', 0.2))
    except ValueError:
        return jsonify({'error': 'Invalid threshold values'}), 400
    
    files = request.files.getlist('file')
    results = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                
                boxes, status = detect_heads(filepath, conf_threshold)
                if status != "Success":
                    results.append({'filename': filename, 'error': status})
                    continue
                
                occluded_boxes = find_occlusions(boxes, iou_threshold)
                
                # Create visualizations
                img = cv2.imread(filepath)
                if img is None:
                    results.append({'filename': filename, 'error': 'Could not process image'})
                    continue
                
                all_img = img.copy()
                occ_img = img.copy()
                
                # Draw all heads
                for i, box in enumerate(boxes):
                    cv2.rectangle(all_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(all_img, str(i + 1), (box[0], box[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw occluded heads
                for i, box in enumerate(boxes):
                    is_occluded = any(calculate_iou(box, boxes[j]) > iou_threshold 
                                    for j in range(i + 1, len(boxes)))
                    if is_occluded:
                        cv2.rectangle(occ_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        cv2.putText(occ_img, str(i + 1), (box[0], box[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(all_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                
                # Save results
                all_filename = f"all_{filename}"
                occ_filename = f"occluded_{filename}"
                cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], all_filename), all_img)
                cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], occ_filename), occ_img)
                
                results.append({
                    'original': filename,
                    'all_heads': all_filename,
                    'occluded_heads': occ_filename,
                    'total_heads': len(boxes),
                    'occluded_count': len(occluded_boxes),
                    'status': 'Success'
                })
                
            except Exception as e:
                results.append({'filename': filename, 'error': f"Processing error: {str(e)}"})
                # Clean up if there was an error
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    return jsonify({'results': results})

@app.route('/delete_image', methods=['POST'])
def delete_image():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    filename = data['filename']
    try:
        files_to_delete = [
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            os.path.join(app.config['RESULT_FOLDER'], f"all_{filename}"),
            os.path.join(app.config['RESULT_FOLDER'], f"occluded_{filename}")
        ]
        
        deleted = False
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted = True
        
        if not deleted:
            return jsonify({'error': 'No files found to delete'}), 404
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
