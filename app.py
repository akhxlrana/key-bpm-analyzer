from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import logging
import traceback
from werkzeug.utils import secure_filename
from audio_analyzer import MusicAnalyzer
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize music analyzer
try:
    analyzer = MusicAnalyzer()
    logger.info("Music analyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize music analyzer: {str(e)}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
    analyzer = None

@app.route('/')
def index():
    logger.info("Index page requested")
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"Error loading page: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("File upload request received")
    
    # Check if analyzer is available
    if analyzer is None:
        logger.error("Music analyzer not initialized - cannot process uploads")
        return jsonify({'error': 'Music analyzer not available. Check server logs for details.'}), 500
    
    if 'file' not in request.files:
        logger.warning("Upload request missing file")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    logger.info(f"File received: {file.filename}")
    
    if file.filename == '':
        logger.warning("Upload request with empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {filepath}")
        
        try:
            file.save(filepath)
            logger.info(f"File saved successfully: {filepath}")
            
            # Analyze the uploaded file
            logger.info(f"Starting analysis of: {filepath}")
            results = analyzer.analyze_song(filepath)
            logger.info(f"Analysis completed. Results: {results}")
            
            if results:
                # Clean up uploaded file
                logger.info(f"Cleaning up file: {filepath}")
                os.remove(filepath)
                
                logger.info("Analysis successful, returning results")
                return jsonify({
                    'success': True,
                    'results': {
                        'key': results['key'],
                        'bpm': results['bpm'],
                        'genre': results['genre']
                    }
                })
            else:
                logger.error("Analysis returned no results")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': 'Failed to analyze audio file'}), 500
                
        except Exception as e:
            # Log detailed error information
            logger.error(f"Exception during file analysis: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                logger.info(f"Cleaning up file after error: {filepath}")
                os.remove(filepath)
            
            error_msg = str(e)
            # Provide specific error messages for common issues
            if "Could not find/load shared object file" in error_msg or "ffmpeg" in error_msg.lower():
                logger.error("FFmpeg not found - this is the likely cause of the error")
                return jsonify({
                    'error': 'Audio processing error: ffmpeg is required to handle MP3 files. Please install ffmpeg and try again.',
                    'solution': 'Install ffmpeg using: conda install -c conda-forge ffmpeg',
                    'detailed_error': error_msg
                }), 500
            elif "DLL initialization routine failed" in error_msg or "llvmlite" in error_msg:
                logger.error("DLL initialization error - librosa dependency issue")
                return jsonify({
                    'error': 'Audio library initialization error. This is a Windows compatibility issue.',
                    'solution': 'Reinstall compatible versions: pip uninstall llvmlite numba librosa -y && pip install librosa==0.10.1 numba==0.58.1 llvmlite==0.41.1',
                    'detailed_error': error_msg
                }), 500
            else:
                logger.error(f"General analysis error: {error_msg}")
                return jsonify({
                    'error': f'Analysis failed: {error_msg}',
                    'detailed_error': error_msg
                }), 500
    
    logger.warning(f"Invalid file type uploaded: {file.filename}")
    return jsonify({'error': 'Invalid file type. Please upload WAV, MP3, FLAC, M4A, or OGG files.'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
