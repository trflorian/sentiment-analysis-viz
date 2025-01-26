from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

data = ["I love you"]

pred = sentiment_pipeline(data)

print(pred)