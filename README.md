# Sentiment Analysis

![demo](https://github.com/user-attachments/assets/baf39a9b-2025-4bd6-9ffe-79cc1ce992dd)

This project is a real-time demo of a sentiment analysis visualization with a smiley based on a text input.
The more positive the sentiment of the input is, the happier the smiley looks and vice versa if the sentiment is really negative.

## How it works

The text input is passed to a sentiment analysis pipeline with a tokenizer and a transformer model pre-trained on twitter data, roughly 58 million tweets.
The specific model used is `cardiffnlp/twitter-roberta-base-sentiment` (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment), which outputs three labels and a confidence score each.
The labels correspond to the classifications *negative*, *neutral* and *positive*. To calculate the final sentiment score in a range of `[-1, 1]`, I simply apply a weighted scaling based on the confidence of each label. 
The negative label prediction confidence score is multiplied by -1, the neutral one by 0 and the positive score by 1.

## Examples

| Positive Sentiment | Negative Sentiment |
| -------- | ------- |
| ![love_is_in_the_air](https://github.com/user-attachments/assets/196b33d0-de47-4f3b-aaad-7d816b622184)  | ![bad weather](https://github.com/user-attachments/assets/d55f4598-065c-4f98-8814-dcd8452a35d5)    |

## Quickstart

### Prerequisites

- [uv](https://docs.astral.sh/uv/) - Package Manager

### Setup

```
uv run sentiment.py
```
