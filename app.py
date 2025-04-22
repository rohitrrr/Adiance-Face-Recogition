from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, g
import os
import uuid
import time
import logging
import datetime
import threading
from werkzeug.utils import secure_filename
import cv2
from functools import wraps
from adiance_wrapper import AdianceWrapper
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def create_app(test_config=None):
    app = Flask(__name__)
    
    
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', os.urandom(24)),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  
        UPLOAD_FOLDER=os.path.join(app.static_folder, 'uploads'),
        ALLOWED_EXTENSIONS={'jpg', 'jpeg', 'png', 'bmp', 'ppm'},
        HF_MODEL_REPO=os.environ.get('HF_MODEL_REPO', 'Rohitrrr/adiance-face-models'),
        ENROLLMENT_DIR=os.environ.get('ENROLLMENT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), "enrollment_db"))
    )
    
    if test_config:
        app.config.update(test_config)
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Setup rate limiting to protect the API
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"]
    )
    
    # Initialize AdianceWrapper as a global object
    adiance = None
    
    # Application statistics
    stats = {
        'enrollments': 0,
        'searches': 0,
        'last_enrollment': None,
        'last_search': None,
        'startup_time': datetime.datetime.now(),
    }
    
    # Middlewares and before request handlers
    @app.before_request
    def before_request():
        g.adiance = get_adiance_instance()
        g.stats = stats
    
    # Get or create AdianceWrapper instance
    def get_adiance_instance():
        nonlocal adiance
        if adiance is None:
            adiance = AdianceWrapper()
            logger.info("AdianceWrapper initialized")
        return adiance
    
    # Custom decorators
    def login_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('logged_in'):
                flash('Please log in to access this page', 'warning')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    
    # Helper functions
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    def save_uploaded_file(file):
        """Securely save uploaded file and return the path"""
        if not file or file.filename == '':
            raise ValueError("No file selected")
            
        if not allowed_file(file.filename):
            raise ValueError(f"Invalid file type. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")
            
        filename = secure_filename(file.filename)
        unique_filename = f"{time.time()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        return file_path
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error.html', 
                              error_code=404, 
                              error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('error.html', 
                              error_code=500, 
                              error_message="Internal server error"), 500
    
    @app.errorhandler(413)
    def request_entity_too_large(e):
        return render_template('error.html', 
                              error_code=413, 
                              error_message="File too large"), 413
    
    # Jinja2 template filters
    @app.template_filter('now')
    def filter_now(format_='%Y'):
        return datetime.datetime.now().strftime(format_)
    
    @app.template_filter('format_date')
    def format_date(date, format_='%Y-%m-%d %H:%M:%S'):
        if date:
            return date.strftime(format_)
        return "N/A"
    
    # Routes
    @app.route('/')
    def index():
        return render_template('index.html', stats=g.stats)
    
    @app.route('/dashboard')
    def dashboard():
        # Get some statistics about enrolled subjects
        enrollment_count = g.adiance.get_enrollment_count() if hasattr(g.adiance, 'get_enrollment_count') else 0
        return render_template('dashboard.html', 
                              stats=g.stats,
                              enrollment_count=enrollment_count)
    
    @app.route('/enroll', methods=['GET', 'POST'])
    def enroll():
        if request.method == 'POST':
            try:
                # Form validation
                if 'file' not in request.files:
                    flash('No file part', 'error')
                    return redirect(request.url)
                
                file = request.files['file']
                subject_id = request.form.get('subject_id', '').strip()
                
                if not subject_id:
                    flash('Subject ID is required', 'error')
                    return redirect(request.url)
                
                # Process file
                try:
                    file_path = save_uploaded_file(file)
                except ValueError as e:
                    flash(str(e), 'error')
                    return redirect(request.url)
                
                # Enroll person using the simplified method
                success, message = g.adiance.enroll_person(file_path, subject_id)
                
                if success:
                    flash(f"Successfully enrolled subject: {subject_id}", 'success')
                    # Update statistics
                    g.stats['enrollments'] += 1
                    g.stats['last_enrollment'] = datetime.datetime.now()
                else:
                    flash(message, 'error')
                    
                # Clean up the uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return redirect(url_for('dashboard'))
                
            except Exception as e:
                logger.exception("Unexpected error during enrollment")
                flash(f"An unexpected error occurred: {str(e)}", 'error')
                return redirect(request.url)
        
        return render_template('enroll.html')
    
    @app.route('/search', methods=['GET', 'POST'])
    def search():
        if request.method == 'POST':
            try:
                # Form validation
                if 'file' not in request.files:
                    flash('No file part', 'error')
                    return redirect(request.url)
                
                file = request.files['file']
                
                # Process file
                try:
                    file_path = save_uploaded_file(file)
                except ValueError as e:
                    flash(str(e), 'error')
                    return redirect(request.url)
                
                # Search for matches using the simplified method
                candidate_count = int(request.form.get('candidate_count', 10))
                success, results = g.adiance.search_person(file_path, candidate_count)
                
                # Clean up the uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                if success:
                    # Update statistics
                    g.stats['searches'] += 1
                    g.stats['last_search'] = datetime.datetime.now()
                    return render_template('search.html', matches=results)
                else:
                    flash(results, 'error')  # Results contains error message
                    return redirect(request.url)
                
            except Exception as e:
                logger.exception("Unexpected error during search")
                flash(f"An unexpected error occurred: {str(e)}", 'error')
                return redirect(request.url)
        
        return render_template('search.html')
    
    @app.route('/batch-enroll', methods=['GET', 'POST'])
    def batch_enroll():
        if request.method == 'POST':
            try:
                # Form validation
                if 'files[]' not in request.files:
                    flash('No files selected', 'error')
                    return redirect(request.url)
                
                files = request.files.getlist('files[]')
                subject_id_prefix = request.form.get('subject_id_prefix', '').strip()
                
                if not subject_id_prefix:
                    flash('Subject ID prefix is required', 'error')
                    return redirect(request.url)
                
                results = []
                for i, file in enumerate(files):
                    try:
                        # Process file
                        file_path = save_uploaded_file(file)
                        
                        # Create and enroll face template
                        template, _ = g.adiance.create_template(file_path)
                        subject_id = f"{subject_id_prefix}_{i+1}"
                        success = g.adiance.enroll_template(template, subject_id)
                        
                        results.append({
                            'filename': file.filename,
                            'subject_id': subject_id,
                            'success': success
                        })
                        
                        # Update statistics if successful
                        if success:
                            g.stats['enrollments'] += 1
                            g.stats['last_enrollment'] = datetime.datetime.now()
                        
                    except Exception as e:
                        logger.error(f"Error processing {file.filename}: {str(e)}")
                        results.append({
                            'filename': file.filename,
                            'subject_id': f"{subject_id_prefix}_{i+1}",
                            'success': False,
                            'error': str(e)
                        })
                    finally:
                        # Clean up the uploaded file
                        if os.path.exists(file_path):
                            os.remove(file_path)
                
                # Finalize enrollment
                g.adiance.finalize_enrollment()
                
                return render_template('batch_results.html', results=results)
            
            except Exception as e:
                logger.exception("Unexpected error during batch enrollment")
                flash(f"An unexpected error occurred: {str(e)}", 'error')
                return redirect(request.url)
        
        return render_template('batch_enroll.html')
    
    @app.route('/api/enroll', methods=['POST'])
    @limiter.limit("20 per minute")
    def api_enroll():
        try:
            # Form validation
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            subject_id = request.form.get('subject_id', '').strip()
            
            if not subject_id:
                return jsonify({'error': 'Subject ID is required'}), 400
            
            # Process file
            try:
                file_path = save_uploaded_file(file)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
            
            # Create and enroll face template
            try:
                template, _ = g.adiance.create_template(file_path)
                success = g.adiance.enroll_template(template, subject_id)
                g.adiance.finalize_enrollment()
                
                # Update statistics
                if success:
                    g.stats['enrollments'] += 1
                    g.stats['last_enrollment'] = datetime.datetime.now()
                
                return jsonify({
                    'success': success,
                    'subject_id': subject_id
                })
                
            except Exception as e:
                logger.error(f"Enrollment error: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                # Clean up the uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        except Exception as e:
            logger.exception("Unexpected error during API enrollment")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/search', methods=['POST'])
    @limiter.limit("20 per minute")
    def api_search():
        try:
            # Form validation
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            
            # Process file
            try:
                file_path = save_uploaded_file(file)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
            
            # Search for matches
            try:
                # Initialize identification
                g.adiance.initialize_identification()
                
                # Create template and search
                template, _ = g.adiance.create_template(file_path)
                candidate_count = int(request.form.get('candidate_count', 10))
                matches = g.adiance.identify(template, candidate_count)
                
                # Update statistics
                g.stats['searches'] += 1
                g.stats['last_search'] = datetime.datetime.now()
                
                return jsonify({
                    'success': True,
                    'matches': [{'template_id': m['template_id'], 'score': m['score']} 
                               for m in matches]
                })
                
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                return jsonify({'error': str(e)}), 500
            finally:
                # Clean up the uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        except Exception as e:
            logger.exception("Unexpected error during API search")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/setup')
    def setup():
        return render_template('setup.html')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
