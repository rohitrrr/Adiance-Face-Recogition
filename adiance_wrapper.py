import os
import ctypes
import numpy as np
import platform
import pickle
from enum import IntEnum
import cv2
import uuid
import logging
import time
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('adiance_wrapper')

# Define necessary enums and structures to match the C++ code
class ReturnCode(IntEnum):
    Success = 0
    ConfigError = 1
    RefuseInput = 2
    ExtractError = 3
    ParseError = 4
    TemplateCreationError = 5
    VerifTemplateError = 6
    FaceDetectionError = 7
    ImageFormatError = 8
    NotImplemented = 9

class Image(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("depth", ctypes.c_int)
    ]

class EyePair(ctypes.Structure):
    _fields_ = [
        ("isLeftEyePresent", ctypes.c_bool),
        ("isRightEyePresent", ctypes.c_bool),
        ("xleft", ctypes.c_float),
        ("yleft", ctypes.c_float),
        ("xright", ctypes.c_float),
        ("yright", ctypes.c_float)
    ]

class Candidate(ctypes.Structure):
    _fields_ = [
        ("isAssigned", ctypes.c_bool),
        ("templateId", ctypes.c_char_p),
        ("similarityScore", ctypes.c_float)
    ]

class AdianceWrapper:
    def __init__(self):
        # Path to library and config files
        app_root = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = os.path.join(app_root, "config")
        self.enrollment_dir = os.path.join(app_root, "enrollment_db")
        
        # Ensure directories exist
        os.makedirs(self.enrollment_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize ONNX Runtime sessions
        self._initialize_models()
        
        # Initialize enrollment database
        self.db_file = os.path.join(self.enrollment_dir, "edb")
        self.manifest_file = os.path.join(self.enrollment_dir, "manifest")
        self.enrollments = {}
        self._load_enrollments()
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.ppm'}
    
    def _load_config(self):
        """Load configuration from adiance.conf"""
        config = {}
        config_path = os.path.join(self.config_dir, "adiance.conf")
        
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}")
            return config
            
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        
        return config
    
    def _initialize_models(self):
        """Initialize RetinaFace and AdaFace models"""
        try:
            # Initialize RetinaFace
            retinaface_path = self.config.get('MTCNN_MODEL_PATH')
            if not retinaface_path or not os.path.isfile(retinaface_path):
                retinaface_path = os.path.join(self.config_dir, "Facedetect.onnx")
                
            logger.info(f"Loading RetinaFace model from: {retinaface_path}")
            self.retinaface_session = ort.InferenceSession(retinaface_path)
            logger.info(f"RetinaFace model input type: {self.retinaface_session.get_inputs()[0].type}")
            
            # Initialize AdaFace
            adaface_path = self.config.get('ADIANCE_MODEL_PATH')
            if not adaface_path or not os.path.isfile(adaface_path):
                adaface_path = os.path.join(self.config_dir, "Embedding.onnx")
                
            logger.info(f"Loading AdaFace model from: {adaface_path}")
            self.adaface_session = ort.InferenceSession(adaface_path)
            logger.info(f"AdaFace model input type: {self.adaface_session.get_inputs()[0].type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _load_enrollments(self):
        """Load existing enrollments from the database"""
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, 'r') as f:
                    for line in f:
                        template_id, template_size, offset = line.strip().split()
                        self.enrollments[template_id] = {
                            'size': int(template_size),
                            'offset': int(offset)
                        }
                logger.info(f"Loaded {len(self.enrollments)} enrollments")
            except Exception as e:
                logger.error(f"Error loading enrollments: {e}")
    
    def _read_image(self, image_path):
        """Read image file with support for multiple formats including PPM"""
        try:
            # Get file extension
            _, ext = os.path.splitext(image_path)
            ext = ext.lower()
            
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {ext}")
            
            # Read image based on format
            if ext == '.ppm':
                # For PPM files, use numpy to read directly
                with open(image_path, 'rb') as f:
                    # Read PPM header
                    magic = f.readline().decode('ascii').strip()
                    if magic != 'P6':
                        raise ValueError("Only binary PPM (P6) is supported")
                    
                    # Read dimensions
                    dims = f.readline().decode('ascii').strip().split()
                    width, height = map(int, dims)
                    
                    # Read max value
                    maxval = int(f.readline().decode('ascii').strip())
                    if maxval != 255:
                        raise ValueError("Only 8-bit PPM is supported")
                    
                    # Read binary data
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                    image = data.reshape((height, width, 3))
                    
                    # Convert from RGB to BGR for OpenCV
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # For other formats, use OpenCV
                image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
            
            # Ensure uint8 type for all images
            return image.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            raise
    
    def detect_faces(self, image):
        """Detect faces using RetinaFace with improved logic matching the C++ implementation"""
        try:
            # Check if image is valid
            if image is None or image.size == 0:
                logger.error("Empty image provided to face detector")
                return []

            # Preprocess image with aspect ratio preservation (like C++ implementation)
            orig_h, orig_w = image.shape[:2]
            target_size = 640
            
            # Preserve aspect ratio during resize (match C++ resizeImage logic)
            if orig_h > orig_w:
                new_h = target_size
                new_w = int(orig_w * (target_size / orig_h))
            else:
                new_w = target_size
                new_h = int(orig_h * (target_size / orig_w))
                
            # Resize and pad
            resized = cv2.resize(image, (new_w, new_h))
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            canvas[:new_h, :new_w] = resized
            
            # Calculate resize factor for later bbox scaling
            resize_factor = new_h / orig_h
            
            # Convert to RGB and normalize as float32
            img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = ((img - np.array([104, 117, 123], dtype=np.float32))
                   / np.float32(255.0)).astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
            
            # Run inference
            input_name = self.retinaface_session.get_inputs()[0].name
            outputs = self.retinaface_session.run(None, {input_name: img})
            
            # Process outputs
            loc, conf, landmarks = outputs
            
            # Use a lower confidence threshold for better recall
            conf_thresh = float(self.config.get('RETINAFACE_CONF_THRESH', 0.3))
            logger.debug(f"Face detection using confidence threshold: {conf_thresh}")
            
            # Decode detections with debug info
            faces = []
            for i in range(loc.shape[1]):
                confidence = conf[0, i, 1]
                
                if confidence > conf_thresh:
                    # Get and scale bounding box to original image dimensions
                    bbox = loc[0, i].copy()
                    
                    # Scale coordinates back to original image dimensions
                    bbox[0] = max(0, int(bbox[0] * target_size / resize_factor))  # x1
                    bbox[1] = max(0, int(bbox[1] * target_size / resize_factor))  # y1
                    bbox[2] = min(orig_w, int(bbox[2] * target_size / resize_factor))  # x2
                    bbox[3] = min(orig_h, int(bbox[3] * target_size / resize_factor))  # y2
                    
                    # Get and scale landmarks
                    lm = landmarks[0, i].reshape(-1, 2).copy()
                    for j in range(lm.shape[0]):
                        lm[j, 0] = lm[j, 0] * target_size / resize_factor  # x
                        lm[j, 1] = lm[j, 1] * target_size / resize_factor  # y
                    
                    face = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'landmarks': lm
                    }
                    faces.append(face)
            
            # Try with lower threshold as fallback if no faces found
            if not faces and conf_thresh > 0.2:
                logger.debug("No faces found, retrying with lower threshold")
                backup_thresh = 0.2
                for i in range(loc.shape[1]):
                    confidence = conf[0, i, 1]
                    if confidence > backup_thresh:
                        # Add face with same logic as above
                        bbox = loc[0, i].copy()
                        bbox[0] = max(0, int(bbox[0] * target_size / resize_factor))
                        bbox[1] = max(0, int(bbox[1] * target_size / resize_factor))
                        bbox[2] = min(orig_w, int(bbox[2] * target_size / resize_factor))
                        bbox[3] = min(orig_h, int(bbox[3] * target_size / resize_factor))
                        
                        lm = landmarks[0, i].reshape(-1, 2).copy()
                        for j in range(lm.shape[0]):
                            lm[j, 0] = lm[j, 0] * target_size / resize_factor
                            lm[j, 1] = lm[j, 1] * target_size / resize_factor
                        
                        face = {
                            'bbox': bbox,
                            'confidence': confidence,
                            'landmarks': lm
                        }
                        faces.append(face)
            
            return faces
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def extract_features(self, face_img):
        """Extract face features using AdaFace"""
        try:
            # Preprocess image
            img = cv2.resize(face_img, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Fix: Explicitly ensure float32 throughout the normalization process
            img = img.astype(np.float32)
            img = ((img - np.float32(127.5)) / np.float32(127.5)).astype(np.float32)
            
            # Ensure contiguous float32 array for transposition
            img = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
            
            # Run inference
            input_name = self.adaface_session.get_inputs()[0].name
            output_name = self.adaface_session.get_outputs()[0].name
            embedding = self.adaface_session.run([output_name], {input_name: img})[0]
            
            # Fix: Normalize while maintaining float32 precision
            embedding = embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (embedding / np.float32(norm)).astype(np.float32)
            
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            raise
    
    def create_template(self, image_path):
        """Create a face template from an image with improved error handling"""
        try:
            # Read image with format support
            image = self._read_image(image_path)
            logger.info(f"Image read successfully: {image.shape}")
            
            # Detect faces
            faces = self.detect_faces(image)
            if not faces:
                # Try with a backup method - sometimes direct extraction works better
                logger.warning("No faces detected, attempting direct feature extraction")
                try:
                    # Try resizing and extracting directly
                    resized_image = cv2.resize(image, (112, 112))
                    features = self.extract_features(resized_image)
                    # Create placeholder landmarks since we don't have actual detection
                    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
                    placeholder_landmarks = np.array([
                        [center_x - 30, center_y - 20],  # left eye
                        [center_x + 30, center_y - 20],  # right eye
                        [center_x, center_y],            # nose
                        [center_x - 20, center_y + 25],  # left mouth
                        [center_x + 20, center_y + 25]   # right mouth
                    ])
                    return features, placeholder_landmarks
                except Exception as backup_err:
                    logger.error(f"Backup method failed: {backup_err}")
                    raise ValueError("No faces detected in the image and backup method failed")
            
            # Get the face with highest confidence
            face = max(faces, key=lambda x: x['confidence'])
            logger.info(f"Selected face with confidence: {face['confidence']:.4f}")
            
            # Extract face region
            bbox = face['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Add safety checks for bbox dimensions
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                logger.warning(f"Invalid bbox: {bbox}, using safe values")
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], max(x1 + 10, x2))
                y2 = min(image.shape[0], max(y1 + 10, y2))
            
            face_img = image[y1:y2, x1:x2]
            logger.info(f"Extracted face region: {face_img.shape}")
            
            # Extract features
            features = self.extract_features(face_img)
            logger.info(f"Extracted features with norm: {np.linalg.norm(features):.4f}")
            
            return features, face['landmarks']
            
        except Exception as e:
            logger.exception(f"Error creating template: {e}")
            raise
    
    def enroll_template(self, template, template_id):
        """Enroll a template in the database"""
        try:
            # Convert template to bytes
            template_bytes = template.tobytes()
            
            # Add to enrollments
            offset = 0
            if self.enrollments:
                last_enrollment = max(self.enrollments.values(), key=lambda x: x['offset'])
                offset = last_enrollment['offset'] + last_enrollment['size']
            
            self.enrollments[template_id] = {
                'size': len(template_bytes),
                'offset': offset
            }
            
            # Write template to database
            with open(self.db_file, 'ab') as f:
                f.write(template_bytes)
            
            # Update manifest
            with open(self.manifest_file, 'a') as f:
                f.write(f"{template_id} {len(template_bytes)} {offset}\n")
            
            logger.info(f"Enrolled template for ID: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error enrolling template: {e}")
            return False
    
    def identify(self, probe_template, candidate_count=10):
        """Identify a probe template against enrolled templates"""
        try:
            matches = []
            
            # Read database file
            with open(self.db_file, 'rb') as f:
                for template_id, info in self.enrollments.items():
                    # Read template
                    f.seek(info['offset'])
                    template_bytes = f.read(info['size'])
                    template = np.frombuffer(template_bytes, dtype=np.float32)
                    
                    # Calculate similarity
                    similarity = np.dot(probe_template, template)
                    
                    matches.append({
                        'template_id': template_id,
                        'score': float(similarity)
                    })
            
            # Sort by similarity score
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top candidates
            return matches[:candidate_count]
            
        except Exception as e:
            logger.error(f"Error during identification: {e}")
            return []
    
    def get_enrollment_count(self):
        """Get the number of enrolled templates"""
        return len(self.enrollments)
    
    def finalize_enrollment(self):
        """Finalize enrollment process by ensuring database files are properly set up"""
        try:
            # Ensure database files exist and are properly formatted
            if not os.path.exists(self.db_file) or not os.path.exists(self.manifest_file):
                # Create empty files if they don't exist
                if not os.path.exists(self.db_file):
                    with open(self.db_file, 'wb') as f:
                        pass
                    
                if not os.path.exists(self.manifest_file):
                    with open(self.manifest_file, 'w') as f:
                        pass
            
            logger.info(f"Enrollment finalized successfully with {len(self.enrollments)} templates")
            return True
        except Exception as e:
            logger.error(f"Error finalizing enrollment: {e}")
            return False

    def initialize_identification(self):
        """Initialize identification by ensuring enrollments are loaded"""
        try:
            # Make sure enrollments are loaded from disk
            if not self.enrollments:
                self._load_enrollments()
                
            # Check if any enrollments exist
            if not self.enrollments:
                logger.warning("No enrollments found in the database")
                
            logger.info(f"Identification initialized with {len(self.enrollments)} templates")
            return True
        except Exception as e:
            logger.error(f"Error initializing identification: {e}")
            return False

    def enroll_person(self, image_path, person_id):
        """Complete end-to-end enrollment process for a person"""
        try:
            # Create template from image
            features, landmarks = self.create_template(image_path)
            
            # Enroll template with person ID
            success = self.enroll_template(features, person_id)
            
            # Finalize enrollment to save changes
            if success:
                self.finalize_enrollment()
                
            return success, "Enrollment successful" if success else "Enrollment failed"
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error enrolling person: {error_msg}")
            return False, f"Error during enrollment: {error_msg}"

    def search_person(self, image_path, candidate_count=10):
        """Complete end-to-end search process for a person"""
        try:
            # Initialize identification
            self.initialize_identification()
            
            # Create template from image
            features, _ = self.create_template(image_path)
            
            # Identify against enrolled templates
            matches = self.identify(features, candidate_count)
            
            return True, matches
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error searching person: {error_msg}")
            return False, f"Error during search: {error_msg}"