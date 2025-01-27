import numpy as np
import cv2

from transformers import pipeline

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="finiteautomata/bertweet-base-sentiment-analysis",
    return_all_scores=True,
)


# Serve the HTML frontend
@app.get("/")
async def get():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>FastAPI Reactive Text Input</title>
        </head>
        <body>
            <h1>Type Something</h1>
            <input type="text" id="textInput" autocomplete="off" />
            <p>Server Response: <span id="response"></span></p>

            <script>
                const ws = new WebSocket("ws://localhost:8000/ws");
                
                ws.onopen = () => {
                    console.log("WebSocket connection established.");
                };

                ws.onmessage = (event) => {
                    document.getElementById("response").innerText = event.data;
                };

                ws.onclose = () => {
                    console.log("WebSocket connection closed.");
                };

                const textInput = document.getElementById("textInput");
                let timeout = null;

                textInput.addEventListener("input", () => {
                    clearTimeout(timeout);
                    // Debounce the input to avoid flooding the server
                    timeout = setTimeout(() => {
                        if (ws.readyState === WebSocket.OPEN) {
                            ws.send(textInput.value);
                        }
                    }, 300);
                });
            </script>
        </body>
    </html>
    """
    return HTMLResponse(html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            pred = sentiment_pipeline(str(data))

            label_scores = {p["label"]: p["score"] for p in pred[0]}

            print(label_scores)

            sentiment_score = label_scores["POS"] - label_scores["NEG"]

            await websocket.send_text(f"Sentiment Score: {sentiment_score:.2f}")
    except WebSocketDisconnect:
        print("Client disconnected")
