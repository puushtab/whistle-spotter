
from huggingface_hub import login
from NatureLM.models import NatureLM

login('MYKEY')

# Download the model from HuggingFace
model = NatureLM.from_pretrained("model_nature")
model = model.eval().to("cuda")


from NatureLM.infer import Pipeline

audio_paths = ["assets/nri-GreenTreeFrogEvergladesNP.mp3"]
queries = ["What is the common name for the focal species in the audio? Answer:"]

pipeline = Pipeline(model=model)

# Run the model over the audio in sliding windows of 10 seconds with a hop length of 10 seconds
results = pipeline(audio_paths, queries, window_length_seconds=10.0, hop_length_seconds=10.0)

print(results)
# ['#0.00s - 10.00s#: Green Treefrog\n']



