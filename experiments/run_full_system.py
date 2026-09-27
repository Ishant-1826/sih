"""
Edge Navigation Engine - API Server
Integrates EKF, Map Matching, and Movement Detection with Live Browser GPS.
"""

import sys
import os
import time
import math
import uvicorn
import asyncio
import json
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from edge_engine.fusion.ekf import EKFNavigationFilter, EKFConfig
from edge_engine.map_matching.hmm import HMMMapMatcher, MovementDetector
from edge_engine.core.data_types import haversine_distance

app = FastAPI(title="IDR-GNSS-FUSION Edge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Edge Engine State
class EdgeEngine:
    def __init__(self):
        self.ekf = EKFNavigationFilter(EKFConfig())
        self.matcher = HMMMapMatcher(search_radius=30.0)
        self.mov_detector = MovementDetector(speed_thresh=0.5, time_thresh=2.0)
        
        self.initialized = False
        self.force_outage = False
        
        # Build a small dummy grid around Delhi for Map Matching demo
        center_lat, center_lon = 28.6139, 77.2090
        for i in range(-5, 5):
            for j in range(-5, 5):
                # Horizontal road
                self.matcher.add_road(
                    (center_lat + i*0.001, center_lon + j*0.001), 
                    (center_lat + i*0.001, center_lon + (j+1)*0.001)
                )
                # Vertical road
                self.matcher.add_road(
                    (center_lat + i*0.001, center_lon + j*0.001), 
                    (center_lat + (i+1)*0.001, center_lon + j*0.001)
                )
                
    def process_live_gps(self, data: dict):
        lat = data.get('lat', 0.0)
        lon = data.get('lon', 0.0)
        speed = data.get('speed', 0.0)
        heading = data.get('heading', 0.0)
        accuracy = data.get('accuracy', 10.0)
        timestamp = data.get('timestamp', time.time())
        
        # 1. GPS Quality & Outlier Filter
        if accuracy > 50.0:
            gnss_valid = False # Reject impossible/terrible GPS
        else:
            gnss_valid = not self.force_outage
            
        # 2. Movement Detection
        dt = 0.1 # assuming 10Hz updates for sim, or use real delta
        mov_state = self.mov_detector.update(speed, accuracy, dt)
        
        # 3. Initialization
        if not self.initialized and lat != 0:
            self.ekf.initialize(lat, lon, heading, speed, timestamp)
            self.initialized = True
            
        if not self.initialized:
            return None
            
        # 4. Fusion / Constraints
        if mov_state == "STATIONARY":
            # CRITICAL STATIONARY BEHAVIOR: Freeze Position, stop DR integration
            # We enforce zero velocity and skip prediction to prevent IMU noise drift
            self.ekf.x[self.ekf.VX] = 0.0
            self.ekf.x[self.ekf.VY] = 0.0
            self.ekf.update_zupt() # Zero velocity update constraint
            mode = "STATIONARY"
        else:
            # Fake IMU for live browser data (in a real app, this comes from deviceMotion)
            # Since browser doesn't easily give IMU in background, we'll feed GPS-derived kinematic prediction to EKF
            self.ekf.step(
                accel_forward=0.0, accel_lateral=0.0, yaw_rate=0.0, dt=dt, timestamp=timestamp,
                gnss_lat=lat if gnss_valid else None,
                gnss_lon=lon if gnss_valid else None,
                gnss_speed=speed if gnss_valid else None,
                gnss_heading=heading if gnss_valid else None,
                gnss_accuracy=accuracy if gnss_valid else None,
                is_stationary=False
            )
            mode = "GNSS_AIDED" if gnss_valid else "DEAD_RECKONING"
            
        state = self.ekf.get_state(timestamp)
        
        # 5. Road-Constrained Map Matching (Project back to road)
        final_lat, final_lon = self.matcher.match(state['latitude'], state['longitude'], state['heading'], state['speed'])
        
        return {
            "lat": final_lat,
            "lon": final_lon,
            "speed": state['speed'],
            "heading": state['heading'],
            "mode": mode,
            "drift": haversine_distance(lat, lon, final_lat, final_lon) if not gnss_valid else 0.0,
            "gt_lat": lat,  # Sending raw GPS as ground truth for visual comparison
            "gt_lon": lon,
            "mov_state": mov_state
        }

engine = EdgeEngine()

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive real raw GPS payload from browser
            message = await websocket.receive_text()
            data = json.loads(message)
            
            # Run Edge Engine Pipeline
            fused_state = engine.process_live_gps(data)
            
            if fused_state:
                await websocket.send_text(json.dumps(fused_state))
                
    except Exception as e:
        print(f"WebSocket closed: {e}")

@app.post("/api/sim/trigger_outage")
def trigger_outage():
    engine.force_outage = True
    return {"status": "Outage Triggered"}

@app.post("/api/sim/recover_gnss")
def recover_gnss():
    engine.force_outage = False
    return {"status": "GNSS Recovered"}

@app.post("/api/sim/set_route")
async def set_route(request: dict):
    coords = request.get("coordinates", [])
    if coords:
        engine.matcher.roads = []
        for i in range(len(coords)-1):
            # GeoJSON gives [lon, lat], matcher expects (lat, lon)
            p1 = (coords[i][1], coords[i][0])
            p2 = (coords[i+1][1], coords[i+1][0])
            engine.matcher.add_road(p1, p2)
    return {"status": "Route set for Map Matching constraints"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
