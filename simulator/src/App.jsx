import { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, useMap, useMapEvents } from 'react-leaflet';
import { Settings, X, Crosshair, Activity, AlertTriangle, ShieldCheck, ChevronDown, Server, Cpu, Navigation2, Radio, Info, Search, Route, Play, PowerOff } from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './index.css';

// --- Icons ---
const createIcon = (color, size=16, border='#fff') => {
  return L.divIcon({
    className: 'custom-icon',
    html: `<div style="background-color: ${color}; width: ${size}px; height: ${size}px; border-radius: 50%; box-shadow: 0 0 12px ${color}; border: 2px solid ${border}; transition: all 0.3s ease;"></div>`,
    iconSize: [size, size],
    iconAnchor: [size/2, size/2]
  });
};

const navIcon = createIcon('#3b82f6', 18); // Blue
const drIcon = createIcon('#f59e0b', 18); // Amber
const statIcon = createIcon('#94a3b8', 16, '#334155'); // Gray
const ipIcon = createIcon('#a855f7', 16, '#fff'); // Purple
const destIcon = createIcon('#ef4444', 20, '#fff'); // Red Destination

// --- Haversine Distance Helper ---
const haversine = (lat1, lon1, lat2, lon2) => {
  const R = 6371e3;
  const p1 = lat1 * Math.PI/180;
  const p2 = lat2 * Math.PI/180;
  const dp = (lat2-lat1) * Math.PI/180;
  const dl = (lon2-lon1) * Math.PI/180;
  const a = Math.sin(dp/2) * Math.sin(dp/2) + Math.cos(p1) * Math.cos(p2) * Math.sin(dl/2) * Math.sin(dl/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
};

// --- Map Controllers ---
function MapController({ lat, lon, isTracking, setIsTracking, firstLockRef, destLat, destLon }) {
  const map = useMap();
  
  useEffect(() => {
    if (typeof lat === 'number' && typeof lon === 'number' && !isNaN(lat) && !isNaN(lon) && lat !== 0 && lon !== 0) {
      if (!firstLockRef.current) {
        map.setView([lat, lon], 16, { animate: false });
        firstLockRef.current = true;
      } else if (isTracking) {
        map.setView([lat, lon], map.getZoom(), { animate: false });
      }
    }
  }, [lat, lon, isTracking, map, firstLockRef]);

  // Fit bounds if destination is set and we're not actively navigating yet
  useEffect(() => {
    if (destLat && destLon && lat && lon) {
       // Only auto-fit once when destination is selected to show route overview
       const bounds = L.latLngBounds([lat, lon], [destLat, destLon]);
       map.fitBounds(bounds, { padding: [50, 50], animate: true });
       setIsTracking(false); // Drop follow lock so user can see full route
    }
  }, [destLat, destLon]); // Deliberately only depend on dest to run once per new destination

  useEffect(() => {
    const handleInteract = () => {
      setIsTracking(false);
    };
    map.on('dragstart', handleInteract);
    map.on('zoomstart', handleInteract);
    map.on('touchstart', handleInteract);
    
    const container = map.getContainer();
    const handleWheel = () => setIsTracking(false);
    container.addEventListener('wheel', handleWheel, { passive: true });
    
    return () => {
      map.off('dragstart', handleInteract);
      map.off('zoomstart', handleInteract);
      map.off('touchstart', handleInteract);
      container.removeEventListener('wheel', handleWheel);
    };
  }, [map, setIsTracking]);
  
  return null;
}

// Allows user to click map to set destination
function MapEvents({ onMapClick }) {
  useMapEvents({
    click(e) {
      onMapClick(e.latlng);
    }
  });
  return null;
}

// --- API Helpers ---
const fetchIp = async () => {
  try {
    const res = await fetch('https://ipwho.is/');
    if (res.ok) {
      const data = await res.json();
      if (data.success && data.latitude && data.longitude) {
        return { 
          lat: parseFloat(data.latitude), 
          lon: parseFloat(data.longitude), 
          city: data.city, region: data.region, country: data.country, 
          source: 'IP', timestamp: Date.now() / 1000
        };
      }
    }
  } catch (e) {}
  
  try {
    const res = await fetch('https://freeipapi.com/api/json');
    if (res.ok) {
      const data = await res.json();
      if (data.latitude && data.longitude) {
        return { 
          lat: parseFloat(data.latitude), 
          lon: parseFloat(data.longitude), 
          city: data.cityName, region: data.regionName, country: data.countryName, 
          source: 'IP', timestamp: Date.now() / 1000
        };
      }
    }
  } catch (e) {}
  
  return { 
    lat: 28.6139, lon: 77.2090, 
    city: "Unknown", region: "Unknown", country: "Unknown", 
    source: 'IP', timestamp: Date.now() / 1000 
  };
};

function useLocationManager(sourceMode, wsRef) {
  const [location, setLocation] = useState(null);
  const [gpsPermission, setGpsPermission] = useState('PROMPT');
  const watchIdRef = useRef(null);
  
  useEffect(() => {
    if (watchIdRef.current) {
      navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }
    let isSubscribed = true;

    const setupLocation = async () => {
      if (sourceMode === 'AUTO' || sourceMode === 'IP') {
        fetchIp().then(ipLoc => {
          if (isSubscribed && ipLoc) {
            setLocation(prev => (!prev || prev.source === 'IP' || sourceMode === 'IP') ? ipLoc : prev);
          }
        });
      }
      if (sourceMode === 'AUTO' || sourceMode === 'GPS') {
        if ("geolocation" in navigator) {
          watchIdRef.current = navigator.geolocation.watchPosition(
            (pos) => {
              if (!isSubscribed) return;
              setGpsPermission('GRANTED');
              const newLoc = {
                lat: pos.coords.latitude, lon: pos.coords.longitude,
                accuracy: pos.coords.accuracy, speed: pos.coords.speed || 0,
                heading: pos.coords.heading || 0, timestamp: pos.timestamp / 1000.0,
                source: 'GPS'
              };
              setLocation(newLoc);
              if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify(newLoc));
              }
            },
            (err) => { if (err.code === err.PERMISSION_DENIED) setGpsPermission('DENIED'); },
            { enableHighAccuracy: true, maximumAge: 0, timeout: 5000 }
          );
        } else {
          setGpsPermission('UNAVAILABLE');
        }
      }
    };
    setupLocation();

    return () => {
      isSubscribed = false;
      if (watchIdRef.current) {
        navigator.geolocation.clearWatch(watchIdRef.current);
        watchIdRef.current = null;
      }
    };
  }, [sourceMode, wsRef]);

  return { location, gpsPermission };
}


function App() {
  const [telemetry] = useState(() => ({ appStart: performance.now() }));
  const [metrics, setMetrics] = useState({ uiRendered: 0, mapRendered: 0, firstLocation: 0, aiReady: 0, backend: 0 });
  
  useEffect(() => {
    setMetrics(m => ({ ...m, uiRendered: performance.now() - telemetry.appStart }));
  }, [telemetry]);

  const [sourceMode, setSourceMode] = useState('AUTO');
  const [showSourceModal, setShowSourceModal] = useState(false);
  const [showDevPanel, setShowDevPanel] = useState(false);
  const [isTracking, setIsTracking] = useState(true);
  
  const wsRef = useRef(null);
  const { location, gpsPermission } = useLocationManager(sourceMode, wsRef);
  const firstLockRef = useRef(false);
  
  const [backendStatus, setBackendStatus] = useState('CONNECTING');
  const [aiStatus, setAiStatus] = useState('INITIALIZING');
  
  // Navigation State Machine
  // States: IDLE, ROUTE_CALCULATING, ROUTE_READY, NAVIGATING, GNSS_OUTAGE, ARRIVED
  const [navStateMachine, setNavStateMachine] = useState('IDLE');
  
  const [destination, setDestination] = useState(null);
  const [routeData, setRouteData] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  const [navState, setNavState] = useState({
    lat: 0, lon: 0, speed: 0, heading: 0,
    mov_state: 'UNCERTAIN', drift: 0, mode: 'WAITING'
  });
  const [fusedPath, setFusedPath] = useState([]);
  
  const latestNavState = useRef(null);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setAiStatus('READY');
      setMetrics(m => ({ ...m, aiReady: performance.now() - telemetry.appStart }));
    }, 1500);
    return () => clearTimeout(timer);
  }, [telemetry]);

  useEffect(() => {
    const connectWs = () => {
      const ws = new WebSocket('ws://localhost:8000/ws/telemetry');
      ws.onopen = () => {
        setBackendStatus('CONNECTED');
        setMetrics(m => ({ ...m, backend: performance.now() - telemetry.appStart }));
      };
      ws.onmessage = (event) => {
        latestNavState.current = JSON.parse(event.data);
      };
      ws.onerror = () => setBackendStatus('OFFLINE');
      ws.onclose = () => {
        setBackendStatus('OFFLINE');
        setTimeout(connectWs, 5000);
      };
      wsRef.current = ws;
      return ws;
    };
    const ws = connectWs();
    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    };
  }, [telemetry]);
  
  // 10 Hz UI Update Loop
  useEffect(() => {
    const interval = setInterval(() => {
      if (latestNavState.current) {
        const data = latestNavState.current;
        if (typeof data.lat === 'number' && !isNaN(data.lat) && data.lat !== 0) {
          setNavState(prev => {
            const newState = { ...prev, ...data };
            // Arrival Detection
            if (destination && (navStateMachine === 'NAVIGATING' || navStateMachine === 'GNSS_OUTAGE')) {
              const distToDest = haversine(newState.lat, newState.lon, destination.lat, destination.lon);
              if (distToDest < 30) {
                setNavStateMachine('ARRIVED');
              }
            }
            return newState;
          });
          setFusedPath(prev => {
            const p = [...prev, [data.lat, data.lon]];
            return p.length > 500 ? p.slice(-500) : p;
          });
        }
        latestNavState.current = null; 
      }
    }, 100);
    return () => clearInterval(interval);
  }, [destination, navStateMachine]);

  useEffect(() => {
    if (location) {
      if (!metrics.firstLocation) setMetrics(m => ({ ...m, firstLocation: performance.now() - telemetry.appStart }));
      if (location.source === 'IP' || navState.mode === 'WAITING') {
        setNavState(prev => ({
          ...prev, lat: location.lat, lon: location.lon, speed: location.speed || 0, heading: location.heading || 0,
          mov_state: location.speed > 1.0 ? 'MOVING' : 'STATIONARY', mode: location.source === 'GPS' ? 'GNSS' : location.source
        }));
      }
    }
  }, [location, metrics.firstLocation, telemetry.appStart, navState.mode]);

  // --- Routing Logic ---
  const calculateRoute = async (destLat, destLon, destName) => {
    if (navState.lat === 0) return; // Need current location
    setNavStateMachine('ROUTE_CALCULATING');
    
    try {
      // OSRM Public API
      const res = await fetch(`https://router.project-osrm.org/route/v1/driving/${navState.lon},${navState.lat};${destLon},${destLat}?overview=full&geometries=geojson`);
      const data = await res.json();
      if (data.code === 'Ok' && data.routes.length > 0) {
        const route = data.routes[0];
        
        // Snap destination to road valid point
        const snappedLon = data.waypoints[1].location[0];
        const snappedLat = data.waypoints[1].location[1];
        
        setDestination({
          lat: snappedLat, lon: snappedLon, name: destName,
          originalLat: destLat, originalLon: destLon
        });
        
        const coords = route.geometry.coordinates; // [lon, lat]
        setRouteData({
          distance: route.distance, // meters
          duration: route.duration, // seconds
          coordinates: coords.map(c => [c[1], c[0]]) // [lat, lon] for Leaflet Polyline
        });
        
        // Pass route to Backend for Map Matching constraints
        fetch('http://localhost:8000/api/sim/set_route', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ coordinates: coords })
        }).catch(() => console.log('Backend not ready for route constraint'));

        setNavStateMachine('ROUTE_READY');
      } else {
        setNavStateMachine('IDLE');
        alert("No valid road route found. Try a different destination.");
      }
    } catch (e) {
      console.error(e);
      setNavStateMachine('IDLE');
      alert("Routing service unavailable. Check network.");
    }
  };

  const handleSearch = async () => {
    if (!searchQuery) return;
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=1`);
      const data = await res.json();
      if (data.length > 0) {
        calculateRoute(parseFloat(data[0].lat), parseFloat(data[0].lon), data[0].display_name.split(',')[0]);
      } else {
        alert("Destination not found.");
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleMapClick = (latlng) => {
    if (navStateMachine === 'IDLE' || navStateMachine === 'ROUTE_READY') {
      calculateRoute(latlng.lat, latlng.lng, "Selected Map Location");
    }
  };
  
  // --- GNSS Simulation ---
  const simulateOutage = () => {
    fetch('http://localhost:8000/api/sim/trigger_outage', { method: 'POST' }).catch(console.error);
    setNavStateMachine('GNSS_OUTAGE');
  };
  
  const restoreGNSS = () => {
    fetch('http://localhost:8000/api/sim/recover_gnss', { method: 'POST' }).catch(console.error);
    setNavStateMachine('NAVIGATING');
  };

  const isValidCenter = typeof navState.lat === 'number' && !isNaN(navState.lat) && navState.lat !== 0;
  const centerPos = isValidCenter ? [navState.lat, navState.lon] : [28.6139, 77.2090];
  const isDR = navState.mode === 'DEAD_RECKONING';
  const isStationary = navState.mov_state === 'STATIONARY';
  
  let currentIcon = statIcon;
  if (navState.mode === 'IP') currentIcon = ipIcon;
  else if (!isStationary) {
     if (isDR) currentIcon = drIcon;
     else currentIcon = navIcon;
  }

  return (
    <div className="app-container">
      <div className="map-layer">
        <MapContainer 
          center={centerPos} 
          zoom={16} 
          style={{ height: '100%', width: '100%' }}
          whenReady={() => setMetrics(m => ({ ...m, mapRendered: performance.now() - telemetry.appStart }))}
        >
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={20} />
          <MapEvents onMapClick={handleMapClick} />
          
          {routeData && (
            <Polyline positions={routeData.coordinates} color="#6366f1" weight={8} opacity={0.7} />
          )}
          {fusedPath.length > 0 && (
            <Polyline positions={fusedPath} color={isDR ? "#f59e0b" : "#3b82f6"} weight={6} opacity={0.8} />
          )}
          {navState.lat !== 0 && (
            <Marker position={[navState.lat, navState.lon]} icon={currentIcon} />
          )}
          {destination && (
            <Marker position={[destination.lat, destination.lon]} icon={destIcon} />
          )}
          
          <MapController lat={centerPos[0]} lon={centerPos[1]} isTracking={isTracking} setIsTracking={setIsTracking} firstLockRef={firstLockRef} destLat={destination?.lat} destLon={destination?.lon} />
        </MapContainer>
      </div>

      <div className="ui-layer">
        
        {/* Search Bar */}
        <div style={{display: 'flex', justifyContent: 'center', marginTop: '20px'}}>
          <div className="interactive-panel" style={{display: 'flex', background: 'var(--panel-bg)', backdropFilter: 'blur(16px)', border: '1px solid var(--panel-border)', borderRadius: '12px', padding: '8px', width: '90%', maxWidth: '500px', boxShadow: 'var(--shadow-lg)'}}>
            <input 
              type="text" 
              placeholder="Search destination or click map..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              style={{flex: 1, background: 'transparent', border: 'none', color: '#fff', padding: '8px 12px', outline: 'none', fontSize: '1rem'}}
            />
            <button className="btn-primary" style={{padding: '8px 16px', borderRadius: '8px'}} onClick={handleSearch}>
              <Search size={18} />
            </button>
          </div>
        </div>

        <div className="panels-container">
          {/* Main Navigation Card */}
          <div className="vehicle-card interactive-panel">
            {navStateMachine === 'IDLE' && navState.lat !== 0 && (
              <div style={{padding: '10px 0', textAlign: 'center', color: 'var(--text-secondary)'}}>
                <Route size={32} style={{margin: '0 auto 10px auto', opacity: 0.5}} />
                <div>Where to?</div>
                <div style={{fontSize: '0.8rem'}}>Search above or click anywhere on the map to set a destination.</div>
              </div>
            )}
            
            {navStateMachine === 'ROUTE_CALCULATING' && (
              <div style={{textAlign: 'center', padding: '20px 0'}}>
                <div style={{fontWeight: 600}}>CALCULATING ROUTE...</div>
              </div>
            )}
            
            {(navStateMachine === 'ROUTE_READY' || navStateMachine === 'NAVIGATING' || navStateMachine === 'GNSS_OUTAGE') && destination && routeData && (
              <>
                <div style={{background: 'rgba(0,0,0,0.3)', padding: '12px', borderRadius: '8px', fontSize: '0.9rem'}}>
                  <div style={{display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px'}}>
                    <div style={{background: '#ef4444', width: '10px', height: '10px', borderRadius: '50%'}}></div>
                    <strong style={{flex: 1}}>{destination.name}</strong>
                  </div>
                  <div style={{color: 'var(--text-secondary)', fontSize: '0.8rem', paddingLeft: '18px'}}>
                    Shortest Road Route
                  </div>
                </div>
                
                <div className="card-row" style={{marginTop: '10px'}}>
                  <div className="metric-group">
                    <span className="metric-val">{(routeData.distance / 1000).toFixed(1)} <span style={{fontSize: '1rem', color: 'var(--text-secondary)'}}>km</span></span>
                    <span className="metric-lbl">DISTANCE</span>
                  </div>
                  <div className="metric-group" style={{textAlign: 'right'}}>
                    <span className="metric-val">{Math.ceil(routeData.duration / 60)} <span style={{fontSize: '1rem', color: 'var(--text-secondary)'}}>min</span></span>
                    <span className="metric-lbl">ETA</span>
                  </div>
                </div>
                <div className="divider"></div>
                
                {navStateMachine === 'ROUTE_READY' && (
                  <button className="btn-primary" style={{justifyContent: 'center', width: '100%'}} onClick={() => { setNavStateMachine('NAVIGATING'); setIsTracking(true); }}>
                    <Play size={18} /> START NAVIGATION
                  </button>
                )}
                
                {navStateMachine === 'NAVIGATING' && (
                  <button className="btn-primary" style={{justifyContent: 'center', width: '100%', background: 'var(--warning-amber)', color: '#000'}} onClick={simulateOutage}>
                    <PowerOff size={18} /> SIMULATE GNSS OUTAGE
                  </button>
                )}
                
                {navStateMachine === 'GNSS_OUTAGE' && (
                  <div style={{background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--alert-red)', padding: '12px', borderRadius: '8px', textAlign: 'center'}}>
                    <div style={{color: 'var(--alert-red)', fontWeight: 'bold', marginBottom: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'}}>
                      <AlertTriangle size={18} /> GNSS OUTAGE
                    </div>
                    <div style={{fontSize: '0.85rem', color: 'var(--text-primary)', marginBottom: '12px'}}>
                      AI DEAD RECKONING ACTIVE
                    </div>
                    <button className="btn-primary" style={{justifyContent: 'center', width: '100%', background: 'var(--ok-green)', padding: '8px'}} onClick={restoreGNSS}>
                      RESTORE GNSS
                    </button>
                  </div>
                )}
              </>
            )}
            
            {navStateMachine === 'ARRIVED' && (
              <div style={{textAlign: 'center', padding: '20px 0'}}>
                <div style={{color: 'var(--ok-green)', fontSize: '1.5rem', fontWeight: 800, marginBottom: '10px'}}>YOU HAVE ARRIVED</div>
                <button className="btn-outline" onClick={() => { setNavStateMachine('IDLE'); setDestination(null); setRouteData(null); setFusedPath([]); }}>
                  Clear Navigation
                </button>
              </div>
            )}
            
            {navState.lat === 0 && (
              <div style={{textAlign: 'center', padding: '20px 0'}}>
                <Navigation2 size={48} color="var(--text-secondary)" style={{margin: '0 auto', marginBottom: '10px', opacity: 0.5}} />
                <div style={{fontWeight: 600}}>SEARCHING FOR LOCATION...</div>
              </div>
            )}
            
            {/* Speed & Location Source footer */}
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px', fontSize: '0.8rem'}}>
               <div style={{display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-secondary)'}}>
                 <Activity size={14} /> {((navState.speed || 0) * 3.6).toFixed(0)} km/h
               </div>
               <div style={{display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-secondary)', cursor: 'pointer'}} onClick={() => setShowSourceModal(true)}>
                 <ShieldCheck size={14} /> {location?.source || '...'}
               </div>
            </div>
          </div>

          <div className="map-controls interactive-panel">
            <button className="icon-btn" onClick={() => setShowDevPanel(!showDevPanel)} title="Diagnostics">
              <Settings size={22} />
            </button>
            <button className={`icon-btn ${isTracking ? 'active' : ''}`} onClick={() => setIsTracking(true)} title="Recenter">
              <Crosshair size={22} />
            </button>
          </div>
        </div>

        {/* --- Source Selection Modal --- */}
        {showSourceModal && (
          <div className="overlay-screen" style={{background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)'}} onClick={() => setShowSourceModal(false)}>
            <div className="interactive-panel" style={{background: 'var(--panel-bg)', padding: '24px', borderRadius: '16px', width: '350px', border: '1px solid var(--panel-border)'}} onClick={e => e.stopPropagation()}>
              <h3 style={{marginTop: 0, marginBottom: '20px', display: 'flex', justifyContent: 'space-between'}}>
                Location Source
                <X size={20} cursor="pointer" onClick={() => setShowSourceModal(false)} />
              </h3>
              
              <div className={`source-option ${sourceMode === 'AUTO' ? 'active' : ''}`} onClick={() => {setSourceMode('AUTO'); setShowSourceModal(false);}}>
                <div style={{display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 'bold'}}>
                  <div className={`radio-dot ${sourceMode === 'AUTO' ? 'active' : ''}`}></div> Auto
                </div>
                <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)', marginLeft: '26px', marginTop: '4px'}}>
                  GPS preferred, IP fallback. Recommended.
                </div>
              </div>

              <div className={`source-option ${sourceMode === 'GPS' ? 'active' : ''}`} onClick={() => {setSourceMode('GPS'); setShowSourceModal(false);}}>
                <div style={{display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 'bold'}}>
                  <div className={`radio-dot ${sourceMode === 'GPS' ? 'active' : ''}`}></div> GPS / Device
                </div>
                <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)', marginLeft: '26px', marginTop: '4px'}}>
                  Most accurate for real navigation.
                </div>
              </div>

              <div className={`source-option ${sourceMode === 'IP' ? 'active' : ''}`} onClick={() => {setSourceMode('IP'); setShowSourceModal(false);}}>
                <div style={{display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 'bold'}}>
                  <div className={`radio-dot ${sourceMode === 'IP' ? 'active' : ''}`}></div> IP Approximate
                </div>
                <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)', marginLeft: '26px', marginTop: '4px'}}>
                  Approximate location only.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* --- Diagnostics Panel --- */}
        {showDevPanel && (
          <div className="dev-panel interactive-panel">
            <div className="dev-panel-header">
              <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
                <Activity size={16} /> DIAGNOSTICS
              </div>
              <X size={18} style={{cursor: 'pointer'}} onClick={() => setShowDevPanel(false)} />
            </div>
            
            <div className="dev-grid">
              <div className="dev-item">
                <span className="dev-lbl">Nav State</span>
                <span className="dev-val">{navStateMachine}</span>
              </div>
              <div className="dev-item">
                <span className="dev-lbl">Engine Mode</span>
                <span className="dev-val">{navState.mode}</span>
              </div>
              <div className="dev-item">
                <span className="dev-lbl">Location Source</span>
                <span className="dev-val">{location ? location.source : 'None'}</span>
              </div>
              <div className="dev-item">
                <span className="dev-lbl">Map Matching</span>
                <span className="dev-val">{navStateMachine !== 'IDLE' ? 'ACTIVE' : 'INACTIVE'}</span>
              </div>
              <div className="dev-item">
                <span className="dev-lbl">AI State</span>
                <span className="dev-val" style={{color: aiStatus === 'READY' ? 'var(--ok-green)' : 'inherit'}}>{aiStatus}</span>
              </div>
              <div className="dev-item">
                <span className="dev-lbl">Backend State</span>
                <span className="dev-val" style={{color: backendStatus === 'CONNECTED' ? 'var(--ok-green)' : 'inherit'}}>{backendStatus}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
