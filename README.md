
<div align="center">
<h1> 

SENTIMENT ANALYSIS

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![GitHub License](https://img.shields.io/github/license/trflorian/sentiment-analysis-viz)

</h1>

![demo](https://github.com/user-attachments/assets/baf39a9b-2025-4bd6-9ffe-79cc1ce992dd)

</div>

This project showcases a real-time sentiment analysis visualization with a smiley based on a text input.
The more positive the sentiment of the input is, the happier the smiley looks and vice versa if the sentiment is really negative.

## How it works

The text input is passed to a sentiment analysis pipeline with a tokenizer and a transformer model pre-trained on twitter data, roughly 58 million tweets.
The specific model used is `cardiffnlp/twitter-roberta-base-sentiment` (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment), which outputs three labels and a confidence score each.
The labels correspond to the classifications *negative*, *neutral* and *positive*. To calculate the final sentiment score in a range of `[-1, 1]`, I simply apply a weighted scaling based on the confidence of each label. 
The negative label prediction confidence score is multiplied by -1, the neutral one by 0 and the positive score by 1.

## Procedural Smiley

Using linear color interpolation in the HSV color space and a simple parabola for the mouth of the smiley, the face is procedurally generated.

![score_smiley_sketch](https://github.com/user-attachments/assets/4e2dd697-fc68-40c6-b4c1-c67f9661f9e4)

## Examples

| Positive Sentiment | Negative Sentiment |
| -------- | ------- |
| ![love_is_in_the_air](https://github.com/user-attachments/assets/196b33d0-de47-4f3b-aaad-7d816b622184)  | ![bad weather](https://github.com/user-attachments/assets/d55f4598-065c-4f98-8814-dcd8452a35d5)    |


## Quickstart

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) as its package/project manager.
You can run `uv sync` in the cloned project folder to get all dependencies setup in a local virtual environment specific for this project.

### Run Demo

To run the project there are currently two applications available:
- An interactve web app using Marimo
- A desktop application using TKinter

#### Interactive Web App

The project includes an interactive web application built with Marimo that provides a user-friendly interface on a webpage. 

```bash
uv run marimo run src/sentiment_analysis/marimo/marimo_app.py
```

![marimo_app](https://github.com/user-attachments/assets/f8b92b65-d26c-4f81-88c3-667a83be720e)


#### Tkinter Desktop App

Make sure that you have all required packages installed for tkinter in order to run the standalone application.

```bash
uv run ctk_app
```

![ctk](https://github.com/user-attachments/assets/379c2e76-c9c0-4187-b70e-e771dd9487ef)

