import librosa
import numpy as np
import os
import logging
import traceback

class MusicAnalyzer:
    def __init__(self):
        # Set up logging for this class
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing MusicAnalyzer")
    
    def extract_features(self, audio_path):
        """Extract audio features for analysis"""
        self.logger.info(f"Starting feature extraction for: {audio_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                self.logger.error(f"Audio file not found: {audio_path}")
                return None
            
            # Log file info
            file_size = os.path.getsize(audio_path)
            self.logger.info(f"File size: {file_size} bytes")
            
            # Load audio file
            self.logger.info("Loading audio file with librosa...")
            y, sr = librosa.load(audio_path)
            self.logger.info(f"Audio loaded successfully. Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")
            
            # Extract features
            features = {}
            
            # BPM (Beats Per Minute)
            self.logger.info("Extracting BPM...")
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            # Add secondary BPM estimation for better accuracy
            tempo_alt = librosa.beat.tempo(y=y, sr=sr)[0]
            # Use the average if they differ significantly, else use primary
            if abs(tempo - tempo_alt) > 20:
                features['bpm'] = float((tempo + tempo_alt) / 2)
                self.logger.info(f"BPM extracted (averaged): {features['bpm']} (primary: {tempo}, alt: {tempo_alt})")
            else:
                features['bpm'] = float(tempo)
                self.logger.info(f"BPM extracted: {features['bpm']}")
            
            # Key detection using chroma features
            self.logger.info("Extracting key using chroma features...")
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            # Define proper key profiles for major keys (rotated versions of C major)
            base_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # C major
            key_profiles = {}
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            for i, key in enumerate(key_names):
                key_profiles[key] = np.roll(base_profile, i).tolist()
            
            # Calculate key correlation
            chroma_mean = np.mean(chroma, axis=1)
            key_scores = {}
            for key, profile in key_profiles.items():
                correlation = np.corrcoef(chroma_mean, profile)[0, 1]
                key_scores[key] = correlation
            
            # Find best key
            best_key = max(key_scores, key=key_scores.get)
            features['key'] = best_key
            self.logger.info(f"Key detected: {best_key}")
            
            # Additional features for genre classification
            self.logger.info("Extracting MFCC features...")
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
            features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
            
            # Spectral features
            self.logger.info("Extracting spectral features...")
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Zero crossing rate
            self.logger.info("Extracting zero crossing rate...")
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # Spectral rolloff
            self.logger.info("Extracting spectral rolloff...")
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['rolloff_mean'] = float(np.mean(rolloff))
            features['rolloff_std'] = float(np.std(rolloff))
            
            self.logger.info("Feature extraction completed successfully")
            return features
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error extracting features: {error_msg}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Provide specific error messages for common issues
            if "Could not find/load shared object file" in error_msg or "ffmpeg" in error_msg.lower():
                self.logger.error("FFmpeg dependency missing!")
                self.logger.error("SOLUTION: Install ffmpeg to handle MP3 files:")
                self.logger.error("  Option 1: conda install -c conda-forge ffmpeg")
                self.logger.error("  Option 2: pip install ffmpeg-python")
                self.logger.error("  Option 3: Download ffmpeg from https://ffmpeg.org/")
            elif "DLL initialization routine failed" in error_msg or "llvmlite" in error_msg:
                self.logger.error("DLL initialization error with librosa dependencies!")
                self.logger.error("SOLUTION: Reinstall with compatible versions:")
                self.logger.error("  pip uninstall llvmlite numba librosa -y")
                self.logger.error("  pip install librosa==0.10.1 numba==0.58.1 llvmlite==0.41.1")
            
            return None
    

    
    def predict_genre(self, features):
        """Predict genre from features using simple heuristics"""
        self.logger.info("Starting genre prediction using heuristics...")

        try:
            bpm = features['bpm']
            spectral_centroid = features['spectral_centroid_mean']
            zcr = features['zcr_mean']

            # Simple heuristic-based genre classification
            if bpm > 140 and spectral_centroid > 3000:
                genre = "Electronic/Dance"
            elif bpm > 120:
                genre = "Pop/Rock"
            elif spectral_centroid < 2000 and zcr < 0.1:
                genre = "Classical"
            elif spectral_centroid > 2500 and bpm < 100:
                genre = "Jazz"
            else:
                genre = "Other"

            self.logger.info(f"Genre predicted: {genre} (BPM: {bpm}, Spectral Centroid: {spectral_centroid:.2f}, ZCR: {zcr:.4f})")
            return genre

        except Exception as e:
            self.logger.error(f"Error predicting genre: {str(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return "Unknown"
    
    def analyze_song(self, audio_path):
        """Complete analysis of a song"""
        self.logger.info(f"Starting complete song analysis for: {audio_path}")
        
        features = self.extract_features(audio_path)
        if features is None:
            self.logger.error("Feature extraction failed, cannot continue analysis")
            return None
        
        # Predict genre
        genre = self.predict_genre(features)
        
        result = {
            'key': features['key'],
            'bpm': round(features['bpm'], 2),
            'genre': genre,
            'features': features
        }
        
        self.logger.info(f"Song analysis completed successfully: Key={result['key']}, BPM={result['bpm']}, Genre={result['genre']}")
        return result
