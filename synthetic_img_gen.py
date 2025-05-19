import cv2
import albumentations as A
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class SyntheticDataGenerator:
    def __init__(self):
        # Configuration
        self.INPUT_IMAGE_DIR = "data/images"
        self.INPUT_LABEL_DIR = "data/labels"
        self.OUTPUT_DIR = "synthetic_data"
        self.GENERATION_MULTIPLIER = 3  # Generate 3x original dataset
        self.OCCLUSION_PROBABILITY = 0.4  # 40% chance to add occlusion
        self.MIN_VISIBILITY = 0.3  # Minimum visible portion of bbox after augmentation
        
        # Supported image extensions
        self.SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        
        # Create output directories
        os.makedirs(os.path.join(self.OUTPUT_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIR, "labels"), exist_ok=True)
        
        # Initialize augmentation pipeline
        self.transform = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self):
        """Create the augmentation pipeline with occlusion handling"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=20, p=0.4),
            A.Blur(blur_limit=3, p=0.1),
            A.CLAHE(p=0.2),
            A.RandomGamma(p=0.2),
            A.RandomShadow(p=0.2),
            A.RandomSunFlare(p=0.1),
        ], bbox_params=A.BboxParams(
            format='albumentations',  # Expects [x_min, y_min, x_max, y_max] in pixels
            min_visibility=self.MIN_VISIBILITY,
            check_each_transform=True
        ))
    
    def _yolo_to_albumentations(self, box, img_width, img_height):
        """Convert YOLO format to Albumentations format (absolute pixel coordinates)"""
        x_center, y_center, width, height = map(float, box)
        
        # Assume YOLO coordinates are normalized
        x_min = (x_center - width / 2) * img_width
        y_min = (y_center - height / 2) * img_height
        x_max = (x_center + width / 2) * img_width
        y_max = (y_center + height / 2) * img_height
        
        # Clip to image boundaries
        x_min = max(0.0, min(x_min, img_width))
        y_min = max(0.0, min(y_min, img_height))
        x_max = max(0.0, min(x_max, img_width))
        y_max = max(0.0, min(y_max, img_height))
        
        return [x_min, y_min, x_max, y_max]
    
    def _albumentations_to_yolo(self, box, img_width, img_height):
        """Convert Albumentations format to YOLO format with normalization"""
        x_min, y_min, x_max, y_max = box
        
        # Clip to image boundaries
        x_min = max(0.0, min(x_min, img_width))
        y_min = max(0.0, min(y_min, img_height))
        x_max = max(0.0, min(x_max, img_width))
        y_max = max(0.0, min(y_max, img_height))
        
        # Convert to YOLO format (normalized)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Clip to [0, 1] range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        return [x_center, y_center, width, height]
    
    def _add_synthetic_occlusion(self, image):
        """Add realistic occlusions to the image"""
        h, w = image.shape[:2]
        
        # Add 1-3 occlusions
        for _ in range(random.randint(1, 3)):
            # Random size between 10-40% of image dimensions
            occlusion_w = random.randint(int(w * 0.1), int(w * 0.4))
            occlusion_h = random.randint(int(h * 0.1), int(h * 0.4))
            
            # Random position
            x = random.randint(0, w - occlusion_w)
            y = random.randint(0, h - occlusion_h)
            
            # Create semi-transparent occlusion
            occlusion = np.zeros((occlusion_h, occlusion_w, 3), dtype=np.uint8)
            
            # Random occlusion type (gray rectangle or noise)
            if random.random() > 0.5:
                occlusion += random.randint(50, 200)  # Gray rectangle
            else:
                occlusion = np.random.randint(0, 255, (occlusion_h, occlusion_w, 3), dtype=np.uint8)  # Noise
            
            # Blend with original image
            alpha = random.uniform(0.5, 0.8)  # Partial transparency
            roi = image[y:y + occlusion_h, x:x + occlusion_w]
            blended = cv2.addWeighted(occlusion, alpha, roi, 1 - alpha, 0)
            image[y:y + occlusion_h, x:x + occlusion_w] = blended
        
        return image
    
    def _process_single_image(self, img_path, iteration):
        """Process a single image and generate augmented versions"""
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to load image: {img_path}")
                return 0
            
            h, w = img.shape[:2]
            label_path = Path(self.INPUT_LABEL_DIR) / f"{img_path.stem}.txt"
            
            # Skip if label file doesn't exist
            if not label_path.exists():
                print(f"Label file not found: {label_path}")
                return 0
            
            # Read YOLO labels
            with open(label_path, 'r') as f:
                yolo_boxes = []
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Invalid label format in {label_path}: {line}")
                        continue
                    yolo_boxes.append([float(x) for x in parts])
            
            # Convert to Albumentations format
            bboxes = []
            for box in yolo_boxes:
                class_id = int(box[0])
                yolo_box = box[1:5]  # [x_center, y_center, width, height]
                
                # Validate YOLO coordinates (should be normalized)
                if any(not (0 <= x <= 1) for x in yolo_box):
                    print(f"Invalid YOLO coordinates in {label_path}: {yolo_box}")
                    continue
                
                alb_bbox = self._yolo_to_albumentations(yolo_box, w, h)
                # Ensure valid bbox
                if alb_bbox[2] > alb_bbox[0] and alb_bbox[3] > alb_bbox[1]:
                    bboxes.append(alb_bbox + [class_id])
            
            # Skip if no valid bounding boxes
            if not bboxes:
                print(f"No valid bounding boxes for {img_path}")
                return 0
            
            # Apply augmentations
            augmented = self.transform(image=img, bboxes=bboxes)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            
            # Clip and validate augmented bounding boxes
            clipped_bboxes = []
            for bbox in aug_bboxes:
                if len(bbox) < 5:
                    continue
                x_min, y_min, x_max, y_max, class_id = bbox
                # Clip to image boundaries
                x_min = max(0.0, min(x_min, w))
                y_min = max(0.0, min(y_min, h))
                x_max = max(0.0, min(x_max, w))
                y_max = max(0.0, min(y_max, h))
                # Ensure valid bbox
                if x_max > x_min and y_max > y_min:
                    clipped_bboxes.append([x_min, y_min, x_max, y_max, class_id])
            
            # Skip if no valid bounding boxes after augmentation
            if not clipped_bboxes:
                print(f"No valid bounding boxes after augmentation for {img_path}")
                return 0
            
            # Add synthetic occlusion
            if random.random() < self.OCCLUSION_PROBABILITY:
                aug_img = self._add_synthetic_occlusion(aug_img)
            
            # Generate unique filename
            output_stem = f"synth_{iteration:02d}_{img_path.stem}_{random.randint(1000, 9999)}"
            output_img_path = Path(self.OUTPUT_DIR) / "images" / f"{output_stem}{img_path.suffix}"
            output_label_path = Path(self.OUTPUT_DIR) / "labels" / f"{output_stem}.txt"
            
            # Save augmented image
            cv2.imwrite(str(output_img_path), aug_img)
            
            # Save augmented labels
            with open(output_label_path, 'w') as f:
                for bbox in clipped_bboxes:
                    yolo_bbox = self._albumentations_to_yolo(bbox[:4], w, h)
                    # Validate YOLO coordinates
                    if all(0 <= x <= 1 for x in yolo_bbox):
                        f.write(f"{int(bbox[4])} {' '.join(map(str, yolo_bbox))}\n")
            
            return 1
        
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            return 0
    
    def generate_synthetic_data(self):
        """Main method to generate synthetic data"""
        image_paths = [p for p in Path(self.INPUT_IMAGE_DIR).glob("*") 
                      if p.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        
        if not image_paths:
            print("No valid images found in input directory!")
            return
        
        print(f"Found {len(image_paths)} images in input directory")
        total_generated = 0
        
        # Use multithreading for faster processing
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for iteration in range(self.GENERATION_MULTIPLIER):
                futures = []
                for img_path in image_paths:
                    futures.append(executor.submit(
                        self._process_single_image, img_path, iteration
                    ))
                
                # Track progress with tqdm
                for future in tqdm(futures, desc=f"Generating batch {iteration + 1}"):
                    total_generated += future.result()
        
        print(f"\nSynthetic data generation complete!")
        print(f"Generated {total_generated} new samples in {self.OUTPUT_DIR}")
        print(f"Original dataset: {len(image_paths)} images")
        print(f"Augmented dataset: {len(list(Path(self.OUTPUT_DIR).glob('images/*')))} images")
        print(f"Labels saved in {self.OUTPUT_DIR}/labels")
        print(f"Images saved in {self.OUTPUT_DIR}/images")
# In your synthetic data generator, add this check:
    def is_duplicate(new_box, existing_boxes, threshold=0.05):
        for box in existing_boxes:
            if (abs(box[1] - new_box[1]) < threshold and \
                abs(box[2] - new_box[2]) < threshold and \
                abs(box[3] - new_box[3]) < threshold and \
                abs(box[4] - new_box[4]) < threshold):
                return True
        return False
if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_synthetic_data()
