# /automathon/users/mig_user2/.cache/kagglehub/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2
from perch_hoplite.zoo import model_configs
import numpy as np

# Input: 5 seconds of silence as mono 32 kHz waveform samples.
waveform = np.zeros(5 * 32000, dtype=np.float32)

# Automatically downloads the model from Kaggle.
# If no GPU, try 'perch_v2_cpu'
model = model_configs.load_model_by_name('perch_v2')

outputs = model.embed(waveform)
# do something with outputs.embeddings and outputs.logits['label']
print("Outputs done!")