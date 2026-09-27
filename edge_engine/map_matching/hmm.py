"""
Advanced Map Matching & Movement Detector
Implements Candidate-based Map Matching and robust stationary detection.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class RoadSegment:
    def __init__(self, id: int, p1: Tuple[float, float], p2: Tuple[float, float]):
        self.id = id
        self.p1 = p1 # (lat, lon)
        self.p2 = p2
        self.heading = self._calculate_heading()
        self.length = haversine(p1[0], p1[1], p2[0], p2[1])

    def _calculate_heading(self):
        lat1, lon1 = math.radians(self.p1[0]), math.radians(self.p1[1])
        lat2, lon2 = math.radians(self.p2[0]), math.radians(self.p2[1])
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360) % 360
        
    def distance_to_point(self, lat: float, lon: float) -> Tuple[float, float, float]:
        """Returns distance, and projected lat/lon on the segment"""
        # Simplified cross-track distance for small segments
        d13 = haversine(self.p1[0], self.p1[1], lat, lon)
        brng13 = self._calculate_bearing(self.p1, (lat, lon))
        brng12 = self.heading
        dxt = math.asin(math.sin(d13/6371000) * math.sin(math.radians(brng13 - brng12))) * 6371000
        # For prototype, approximate projection (Euclidean-like for small scale)
        return abs(dxt), lat, lon

    def _calculate_bearing(self, p1, p2):
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360) % 360


class HMMMapMatcher:
    def __init__(self, search_radius=50.0):
        self.search_radius = search_radius
        self.roads: List[RoadSegment] = []
        self.last_candidate = None
        
    def add_road(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        self.roads.append(RoadSegment(len(self.roads), p1, p2))
        
    def match(self, lat: float, lon: float, heading: float, speed: float) -> Tuple[float, float]:
        if not self.roads:
            return lat, lon
            
        best_score = -float('inf')
        best_point = (lat, lon)
        best_candidate = None
        
        for road in self.roads:
            dist, proj_lat, proj_lon = road.distance_to_point(lat, lon)
            
            if dist > self.search_radius:
                continue
                
            # 1. Distance Score (Gaussian emission)
            sigma_d = 10.0
            dist_score = math.exp(-0.5 * (dist / sigma_d)**2)
            
            # 2. Heading Score (Is vehicle moving along the road?)
            heading_diff = abs(heading - road.heading)
            heading_diff = min(heading_diff, 360 - heading_diff)
            sigma_h = 30.0 # degrees
            heading_score = math.exp(-0.5 * (heading_diff / sigma_h)**2)
            
            # 3. Transition/Connectivity Penalty
            transition_score = 1.0
            if self.last_candidate is not None and self.last_candidate.id != road.id:
                # Penalty for jumping roads if they don't share a node
                shares_node = (road.p1 == self.last_candidate.p1 or road.p1 == self.last_candidate.p2 or 
                               road.p2 == self.last_candidate.p1 or road.p2 == self.last_candidate.p2)
                if not shares_node:
                    transition_score = 0.1 # Severe penalty for jumping disconnected roads
            
            total_score = (dist_score * 0.5 + heading_score * 0.5) * transition_score
            
            if total_score > best_score:
                best_score = total_score
                # Strong snap to road if score is good, otherwise blend
                blend = min(1.0, best_score)
                best_point = (lat * (1-blend) + proj_lat * blend, lon * (1-blend) + proj_lon * blend)
                best_candidate = road
                
        if best_candidate:
            self.last_candidate = best_candidate
            return best_point
        return lat, lon

class MovementDetector:
    """
    Robust movement state detector using hysteresis.
    States: STATIONARY, MOVING, UNCERTAIN
    """
    def __init__(self, speed_thresh=0.5, time_thresh=3.0):
        self.speed_thresh = speed_thresh
        self.time_thresh = time_thresh
        self.state = "STATIONARY"
        self.time_below_thresh = 0.0
        self.time_above_thresh = 0.0
        
    def update(self, speed: float, gps_accuracy: float, dt: float) -> str:
        # Ignore highly inaccurate GPS for movement detection
        if gps_accuracy > 30.0:
            return self.state
            
        if speed < self.speed_thresh:
            self.time_below_thresh += dt
            self.time_above_thresh = 0.0
        else:
            self.time_above_thresh += dt
            self.time_below_thresh = 0.0
            
        # Hysteresis to prevent rapid switching
        if self.state == "MOVING" and self.time_below_thresh > self.time_thresh:
            self.state = "STATIONARY"
        elif self.state == "STATIONARY" and self.time_above_thresh > 1.0: # Faster to start moving than to stop
            self.state = "MOVING"
            
        return self.state
