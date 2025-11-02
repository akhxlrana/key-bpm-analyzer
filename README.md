# Music Analyzer - Key, BPM & Genre Detection

A simple web application that analyzes uploaded music files to detect their key, BPM (beats per minute), and genre using machine learning.

## Features

- 🎵 **Key Detection**: Identifies the musical key of the uploaded song
- 🥁 **BPM Analysis**: Calculates the tempo (beats per minute)
- 🎭 **Genre Classification**: Predicts the music genre using ML
- 🌐 **Web Interface**: Easy-to-use drag-and-drop interface
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Supported File Formats

- WAV
- MP3
- FLAC
- M4A
- OGG

## Installation

1. **Install Python** (3.8 or higher) if not already installed
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to: `http://localhost:5000`

3. **Upload a music file** by:
   - Clicking the upload area and selecting a file, or
   - Dragging and dropping a file onto the upload area

4. **Click "Analyze Music"** to get the results

## How It Works

- **Key Detection**: Uses chroma features and correlation with key profiles
- **BPM Analysis**: Employs librosa's beat tracking algorithm
- **Genre Classification**: Uses a Random Forest classifier trained on audio features like MFCC, spectral features, and tempo

## Technical Details

- **Backend**: Flask web framework
- **Audio Processing**: librosa library for audio analysis
- **Machine Learning**: scikit-learn for genre classification
- **Frontend**: HTML5, CSS3, JavaScript with drag-and-drop support

## File Structure

```
music-analyzer/
├── app.py                 # Main Flask application
├── audio_analyzer.py      # Audio analysis and ML logic
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── uploads/               # Temporary file storage (auto-created)
└── README.md              # This file
```

## Notes

- The genre classifier uses a simplified model for demonstration. For production use, train on a large dataset of labeled music.
- Maximum file size is 16MB
- Uploaded files are automatically deleted after analysis
- The application runs on `http://localhost:5000` by default

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
- **Audio format issues**: Ensure your audio file is in a supported format
- **Large files**: The app has a 16MB file size limit
- **Port conflicts**: If port 5000 is busy, modify the port in `app.py`

## Future Enhancements

- Support for more audio formats
- Batch file processing
- More detailed audio analysis
- Export results to CSV/JSON
- Real-time audio analysis
