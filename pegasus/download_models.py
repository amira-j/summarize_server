from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import pipeline
import torch


model = pipeline("summarization", model="google/pegasus-xsum", tokenizer="google/pegasus-xsum", framework="pt", device=2)
model2 = pipeline("summarization", framework="pt", device=2)
model3 = pipeline("summarization", model="lidiya/bart-large-xsum-samsum", framework="pt", device=2)

torch.save(model, "model/pegasus_pipeline.pt")
torch.save(model2, "model/default_pipeline.pt")
torch.save(model3, "model/bart_pipeline.pt")



