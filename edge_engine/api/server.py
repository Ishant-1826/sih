from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import os

app = FastAPI(title="IDR-GNSS-FUSION Edge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state for simulation
current_state = {
    "lat": 28.6139,
    "lon": 77.2090,
    "speed": 0.0,
    "heading": 0.0,
    "mode": "GNSS_AIDED",
    "drift": 0.0
}

@app.get("/api/state")
def get_state():
    return current_state

@app.post("/api/state")
def update_state(state: dict):
    global current_state
    current_state.update(state)
    return {"status": "success"}

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Broadcast state at 10Hz
            await websocket.send_text(json.dumps(current_state))
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"WebSocket Error: {e}")
