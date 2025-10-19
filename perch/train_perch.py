#!/usr/bin/env python3
"""
Training script for European Bee-eater detection using Perch embeddings.

This script:
1. Loads the Perch v2 model
2. Extracts embeddings from labeled bird song samples
3. Trains a binary classifier to detect European Bee-eater (#eubeat1)
4. Saves the trained model for inference

Based on Google's Perch model: https://www.kaggle.com/models/google/bird-vocalization-classifier
"""

import os
import json
import pickle
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Perch model imports
from perch_hoplite.zoo import model_configs


class PerchBeeEaterTrainer:
    """Train a classifier to detect European Bee-eater using Perch embeddings."""
    
    def __init__(self, dataset_path, target_species='eubeat1', sample_rate=32000, segment_length=5.0):
        """
        Initialize the trainer.
        
        Args:
            dataset_path: Path to dataset/bird_songs directory
            target_species: Target species folder name (default: 'eubeat1')
            sample_rate: Sample rate for audio (Perch uses 32kHz)
            segment_length: Length of audio segments in seconds (Perch uses 5s)
        """
        self.dataset_path = Path(dataset_path)
        self.target_species = target_species
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(sample_rate * segment_length)
        
        # Load Perch model
        print("Loading Perch v2 model...")
        self.perch_model = model_configs.load_model_by_name('perch_v2')
        print("Perch model loaded successfully!")
        
        # Classifier components
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=100.0,               # ⭐ Changed from 1.0
            solver='saga',         # ⭐ Changed from 'lbfgs'
            max_iter=2000,        # ⭐ Changed from 1000
            class_weight='balanced',
            random_state=42,
            n_jobs=-1             # ⭐ Added
        )
        
    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        try:
            # Load audio at target sample rate
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Peak normalize to 0.25 (recommended by Perch documentation)
            if np.max(np.abs(audio)) > 0:
                audio = audio * (0.25 / np.max(np.abs(audio)))
            
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_embeddings_from_audio(self, audio):
        """
        Extract Perch embeddings from audio using sliding window.
        
        Args:
            audio: Audio waveform (mono, 32kHz)
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Slide 5-second windows across the audio
        hop_size = self.segment_samples // 2  # ⭐ 50% overlap        
        
        for start in range(0, len(audio) - self.segment_samples + 1, hop_size):
            segment = audio[start:start + self.segment_samples]
            
            # Ensure segment is exactly the right length
            if len(segment) == self.segment_samples:
                # Get Perch embedding
                outputs = self.perch_model.embed(segment)
                # Flatten embedding to 1D vector
                embedding = np.array(outputs.embeddings).flatten()
                embeddings.append(embedding)
        
        return embeddings
    
    def load_species_data(self, species_folder, label):
        """
        Load all audio files from a species folder and extract embeddings.
        
        Args:
            species_folder: Path to species folder
            label: Binary label (1 for target species, 0 for others)
            
        Returns:
            embeddings: List of embedding vectors
            labels: List of corresponding labels
        """
        embeddings = []
        labels = []
        
        species_path = self.dataset_path / species_folder
        audio_files = list(species_path.glob('*.ogg'))
        
        print(f"Processing {species_folder}: {len(audio_files)} files")
        
        for audio_file in tqdm(audio_files, desc=f"  {species_folder}"):
            audio = self.load_audio(audio_file)
            if audio is None:
                continue
            
            # Extract embeddings from this audio file
            file_embeddings = self.extract_embeddings_from_audio(audio)
            
            # Add to dataset
            embeddings.extend(file_embeddings)
            labels.extend([label] * len(file_embeddings))
        
        return embeddings, labels
    
    def prepare_dataset(self):
        """Prepare training dataset from all species folders."""
        all_embeddings = []
        all_labels = []
        
        # Get all species folders
        species_folders = [f.name for f in self.dataset_path.iterdir() if f.is_dir()]
        
        print(f"\nFound {len(species_folders)} species folders")
        print(f"Target species: {self.target_species}")
        
        # Process each species
        for species_folder in species_folders:
            # Label: 1 for target species, 0 for others
            label = 1 if species_folder == self.target_species else 0
            
            embeddings, labels = self.load_species_data(species_folder, label)
            all_embeddings.extend(embeddings)
            all_labels.extend(labels)
        
        # Convert to numpy arrays
        X = np.array(all_embeddings)
        y = np.array(all_labels)
        
        print(f"\nDataset prepared:")
        print(f"  Total segments: {len(y)}")
        print(f"  Positive samples (Bee-eater): {np.sum(y == 1)}")
        print(f"  Negative samples (other species): {np.sum(y == 0)}")
        print(f"  Embedding dimension: {X.shape[1]}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """Train the classifier."""
        print("\n" + "="*60)
        print("Training Binary Classifier")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {len(y_train)} samples")
        print(f"Test set: {len(y_test)} samples")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        print("Training logistic regression classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        
        y_pred = self.classifier.predict(X_test_scaled)
        y_pred_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Other Species', 'Bee-eater']))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        return X_train, X_test, y_train, y_test, y_pred_proba
    
    def save_model(self, output_path):
        """Save the trained model components."""
        model_data = {
            'scaler': self.scaler,
            'classifier': self.classifier,
            'sample_rate': self.sample_rate,
            'segment_length': self.segment_length,
            'target_species': self.target_species
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {output_path}")


def main():
    """Main training pipeline."""
    # Paths
    dataset_path = Path(__file__).parent.parent / 'dataset' / 'bird_songs'
    model_output_path = Path(__file__).parent.parent / 'models' / 'perch_beeeater_classifier2.pkl'
    
    print("="*60)
    print("European Bee-eater Detection - Training Pipeline")
    print("="*60)
    print(f"\nDataset path: {dataset_path}")
    print(f"Model output: {model_output_path}")
    
    # Initialize trainer
    trainer = PerchBeeEaterTrainer(dataset_path)
    
    # Prepare dataset
    print("\n" + "="*60)
    print("Step 1: Extracting Embeddings from Training Data")
    print("="*60)
    X, y = trainer.prepare_dataset()
    
    # Train classifier
    print("\n" + "="*60)
    print("Step 2: Training Classifier")
    print("="*60)
    trainer.train(X, y)
    
    # Save model
    print("\n" + "="*60)
    print("Step 3: Saving Model")
    print("="*60)
    trainer.save_model(model_output_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nYou can now use the model for predictions:")
    print(f"  python predict_perch.py")


if __name__ == '__main__':
    main()
