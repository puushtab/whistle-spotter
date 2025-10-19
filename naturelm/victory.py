import os
import json
from pathlib import Path
from NatureLM.models import NatureLM
from NatureLM.infer import Pipeline
from huggingface_hub import login
import librosa
import soundfile as sf

# Login to HuggingFace
login('MY_KEY')

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Load the NatureLM model
model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
model = model.eval().to("cuda")

# Initialize the inference pipeline
pipeline = Pipeline(model=model)

# Paths
predictions_file = "/automathon/users/mig_user2/Automathon_Sujet/predictions_perch.json"
dataset_dir = "/automathon/users/mig_user2/Automathon_Sujet/dataset_validation"
output_file = "/automathon/users/mig_user2/Automathon_Sujet/refined_predictions.json"

# Load predictions_perch.json
with open(predictions_file, "r") as f:
    predictions = json.load(f)

# Refine predictions
refined_predictions = {"audios": {}}

for audio_id, data in predictions["audios"].items():
    audio_path = Path(dataset_dir) / f"{audio_id}.ogg"
    if not audio_path.exists():
        print(f"Audio file {audio_path} not found. Skipping...")
        continue

    print(f"Processing audio: {audio_path}")
    refined_timestamps = []

    # Load audio
    audio, sr = sf.read(audio_path)
    audio = audio.T[0] if audio.ndim > 1 else audio  # Use the first channel if stereo

    # Process each interval
    for interval in data.get("timestamps", []):
        if len(interval) < 2:
            continue  # Skip invalid intervals

        start, end = interval[0], interval[-1]
        segment = audio[int(start * sr):int(end * sr)]

        # Save the segment as a temporary file
        segment_path = f"/tmp/{audio_id}_segment.ogg"
        sf.write(segment_path, segment, sr)

        # Run NatureLM on the segment with sliding windows
        results = pipeline(
            [segment_path],
            ["What is the common name for the bird specie in the audio? Answer:"],
            window_length_seconds=1.0,
            hop_length_seconds=1.0,
            verbose=False
        )

        # Parse results to refine intervals
        for i, result in enumerate(results):
            if "European Bee Eater" in result:  # Check for the specific phrase
                refined_timestamps.append([start + i, start + i + 1])

    # Save refined intervals
    refined_predictions["audios"][audio_id] = {
        "id": audio_id,
        "timestamps": refined_timestamps
    }

# Save refined predictions to a new JSON file
with open(output_file, "w") as f:
    json.dump(refined_predictions, f, indent=4)

print(f"Refined predictions saved to {output_file}")
