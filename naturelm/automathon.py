
# %%

from huggingface_hub import login
from NatureLM.models import NatureLM
import os

login('MYKEY')

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Download the model from HuggingFace
model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
model = model.eval().to("cuda")


# %%
from NatureLM.infer import Pipeline

audio_paths = ["../../dataset/parc_audios/data/audio_7.ogg"]
queries = ["What is the common name for the bird specie in the audio? Answer:"]

pipeline = Pipeline(model=model)

# Run the model over the audio in sliding windows of 10 seconds with a hop length of 10 seconds
results = pipeline(audio_paths, queries, window_length_seconds=10.0, hop_length_seconds=10.0, verbose=True)

print(results)
# ['#0.00s - 10.00s#: Green Treefrog\n']



