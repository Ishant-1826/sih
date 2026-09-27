import { useState, useEffect, useRef, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, useMap, useMapEvents } from 'react-leaflet';
import { 
  Settings, X, Crosshair, Activity, AlertTriangle, ShieldCheck, Search, 
  Play, PowerOff, Home, Briefcase, Star, Clock, MapPin, Map, Volume2, Moon, Sun, Monitor 
} from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './index.css';

import StorageService from './services/StorageService';
import VoiceService from './services/VoiceService';
import { haversine, parseOSRMInstruction, generateVoiceAnnouncement } from './utils/geo';

// --- Icons ---
const createIcon = (color, size=16, border='#fff') => L.divIcon({
  className: 'custom-icon',
  html: `<div style="background-color: ${color}; width: ${size}px; height: ${size}px; border-radius: 50%; box-shadow: 0 0 12px ${color}; border: 2px solid ${border}; transition: all 0.3s ease;"></div>`,
  iconSize: [size, size], iconAnchor: [size/2, size/2]
});

const navIcon = createIcon('#3b82f6', 18);
const drIcon = createIcon('#f59e0b', 18);
const statIcon = createIcon('#94a3b8', 16, '#334155');
const ipIcon = createIcon('#a855f7', 16, '#fff');
const destIcon = createIcon('#ef4444', 20, '#fff');

// --- Map Controllers ---
function MapController({ lat, lon, isTracking, setIsTracking, firstLockRef, destLat, destLon }) {
  const map = useMap();
  
  useEffect(() => {
    if (typeof lat === 'number' && typeof lon === 'number' && lat !== 0) {
      if (!firstLockRef.current) {
        map.setView([lat, lon], 16, { animate: false });
        firstLockRef.current = true;
      } else if (isTracking) {
        map.setView([lat, lon], map.getZoom(), { animate: false });
      }
    }
  }, [lat, lon, isTracking, map, firstLockRef]);

  useEffect(() => {
    if (destLat && destLon && lat && lon) {
       const bounds = L.latLngBounds([lat, lon], [destLat, destLon]);
       map.fitBounds(bounds, { padding: [50, 50], animate: true });
       setIsTracking(false);
    }
  }, [destLat, destLon]);

  useEffect(() => {
    const handleInteract = () => setIsTracking(false);
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

function MapEvents({ onMapClick }) {
  useMapEvents({ click(e) { onMapClick(e.latlng); } });
  return null;
}

// --- App Component ---
function App() {
  // Theme & Settings
  const [theme, setTheme] = useState(() => StorageService.get('theme', 'system'));
  const [voiceEnabled, setVoiceEnabled] = useState(() => StorageService.get('voiceEnabled', true));
  
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
    } else {
      root.setAttribute('data-theme', theme);
    }
    StorageService.set('theme', theme);
  }, [theme]);

  useEffect(() => {
    VoiceService.setEnabled(voiceEnabled);
    StorageService.set('voiceEnabled', voiceEnabled);
  }, [voiceEnabled]);

  // Core State
  const [telemetry] = useState(() => ({ appStart: performance.now() }));
  const [isTracking, setIsTracking] = useState(true);
  const firstLockRef = useRef(false);
  
  const [navStateMachine, setNavStateMachine] = useState('IDLE');
  const [destination, setDestination] = useState(null);
  const [routeData, setRouteData] = useState(null);
  const [navState, setNavState] = useState({ lat: 0, lon: 0, speed: 0, heading: 0, mov_state: 'WAITING', mode: 'WAITING' });
  const [fusedPath, setFusedPath] = useState([]);
  const wsRef = useRef(null);
  const latestNavState = useRef(null);

  // Search & Places
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [showSearch, setShowSearch] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  
  const [recentPlaces, setRecentPlaces] = useState(() => StorageService.getRecent());
  const [savedPlaces, setSavedPlaces] = useState(() => StorageService.getSaved());

  // Save Modal State
  const [saveModal, setSaveModal] = useState({ open: false, mode: 'create', place: null, label: '', warning: '', warningType: '', existingId: null });


  // Voice tracking
  const voiceStateRef = useRef({ announcedId: null, lastAnnouncedStage: null });

  // Debounced Search
  useEffect(() => {
    if (!searchQuery) { setSearchResults([]); return; }
    const timeout = setTimeout(async () => {
      try {
        const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=5&addressdetails=1`);
        const data = await res.json();
        setSearchResults(data);
      } catch (e) {}
    }, 400);
    return () => clearTimeout(timeout);
  }, [searchQuery]);

  // Backend WS
  useEffect(() => {
    const connectWs = () => {
      const ws = new WebSocket('ws://localhost:8000/ws/telemetry');
      ws.onmessage = (event) => { latestNavState.current = JSON.parse(event.data); };
      ws.onclose = () => setTimeout(connectWs, 5000);
      wsRef.current = ws;
      return ws;
    };
    const ws = connectWs();
    return () => { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); };
  }, []);

  // Update Loop & Navigation Logic
  useEffect(() => {
    const interval = setInterval(() => {
      if (latestNavState.current) {
        const data = latestNavState.current;
        if (typeof data.lat === 'number' && !isNaN(data.lat) && data.lat !== 0) {
          setNavState(prev => {
            const newState = { ...prev, ...data };
            
            // Route Progress & Voice Logic
            if (destination && routeData && (navStateMachine === 'NAVIGATING' || navStateMachine === 'GNSS_OUTAGE')) {
              const distToDest = haversine(newState.lat, newState.lon, destination.lat, destination.lon);
              
              if (distToDest < 30) {
                if (navStateMachine !== 'ARRIVED') {
                  VoiceService.speak("You have arrived at your destination.");
                  setNavStateMachine('ARRIVED');
                }
              } else if (routeData.steps) {
                // Find current step
                let closestStep = routeData.steps[0];
                let minStepDist = Infinity;
                let nextStepIndex = 0;
                
                for (let i = 0; i < routeData.steps.length - 1; i++) {
                  const stepLoc = routeData.steps[i+1].maneuver.location;
                  const d = haversine(newState.lat, newState.lon, stepLoc[1], stepLoc[0]);
                  if (d < minStepDist) { minStepDist = d; closestStep = routeData.steps[i+1]; nextStepIndex = i+1; }
                }

                // If next turn is within 1000m, handle voice
                if (minStepDist < 1000 && minStepDist > 10) {
                  let stage = 'far';
                  if (minStepDist < 50) stage = 'immediate';
                  else if (minStepDist < 200) stage = 'near';
                  
                  const instructionId = `step_${nextStepIndex}`;
                  const { announcedId, lastAnnouncedStage } = voiceStateRef.current;
                  
                  if (announcedId !== instructionId || lastAnnouncedStage !== stage) {
                    const instructionText = parseOSRMInstruction(closestStep);
                    const announcement = generateVoiceAnnouncement(instructionText, minStepDist);
                    VoiceService.speak(announcement);
                    voiceStateRef.current = { announcedId: instructionId, lastAnnouncedStage: stage };
                  }
                }
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
  }, [destination, routeData, navStateMachine]);

  // Initial GPS Fetch
  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.watchPosition(
        (pos) => {
          const loc = {
            lat: pos.coords.latitude, lon: pos.coords.longitude,
            speed: pos.coords.speed || 0, heading: pos.coords.heading || 0,
            timestamp: pos.timestamp / 1000.0, source: 'GPS'
          };
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(loc));
          }
          if (navState.mode === 'WAITING') setNavState(p => ({ ...p, lat: loc.lat, lon: loc.lon }));
        },
        () => {}, { enableHighAccuracy: true }
      );
    }
  }, [navState.mode]);

  // --- Routing ---
  const calculateRoute = async (destLat, destLon, destName, category="saved", existingPlace=null) => {
    if (navState.lat === 0) return;
    setNavStateMachine('ROUTE_CALCULATING');
    setShowSearch(false);
    
    if (existingPlace && existingPlace.id) {
        StorageService.updateLastUsed(existingPlace.id);
        setSavedPlaces(StorageService.getSaved());
    }
    
    try {
      const res = await fetch(`https://router.project-osrm.org/route/v1/driving/${navState.lon},${navState.lat};${destLon},${destLat}?overview=full&geometries=geojson&steps=true`);
      const data = await res.json();
      if (data.code === 'Ok' && data.routes.length > 0) {
        const route = data.routes[0];
        const snappedLon = data.waypoints[1].location[0];
        const snappedLat = data.waypoints[1].location[1];
        
        const place = existingPlace ? { ...existingPlace, lat: snappedLat, lon: snappedLon } : { lat: snappedLat, lon: snappedLon, name: destName, type: category };

        setDestination(place);
        StorageService.addRecent(place);
        setRecentPlaces(StorageService.getRecent());
        
        setRouteData({
          distance: route.distance,
          duration: route.duration,
          coordinates: route.geometry.coordinates.map(c => [c[1], c[0]]),
          steps: route.legs[0].steps
        });
        
        fetch('http://localhost:8000/api/sim/set_route', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ coordinates: route.geometry.coordinates })
        }).catch(()=>{});

        setNavStateMachine('ROUTE_READY');
      } else {
        setNavStateMachine('IDLE');
        alert("No route found.");
      }
    } catch (e) {
      setNavStateMachine('IDLE');
      alert("Routing unavailable.");
    }
  };

  const handleStartNav = () => {
    setNavStateMachine('NAVIGATING');
    setIsTracking(true);
    VoiceService.speak("Navigation started.");
  };

  const openSaveModal = (place, mode = 'create') => {
    setSaveModal({
      open: true,
      mode,
      place,
      label: mode === 'edit' ? place.label : '',
      warning: '', warningType: '', existingId: null
    });
  };

  const handleConfirmSave = (force = false) => {
    const label = saveModal.label.trim();
    if (!label) {
      setSaveModal(p => ({ ...p, warning: 'Please enter a name.' }));
      return;
    }
    
    if (!force) {
        if (saveModal.mode === 'create' || (saveModal.mode === 'edit' && label.toLowerCase() !== saveModal.place.label?.toLowerCase())) {
            const existingName = savedPlaces.find(p => p.label.toLowerCase() === label.toLowerCase());
            if (existingName) {
                setSaveModal(p => ({ ...p, warning: 'A place with this name already exists.', warningType: 'name' }));
                return;
            }
        }
        
        if (saveModal.mode === 'create') {
            const existingCoords = savedPlaces.find(p => Math.abs(p.lat - saveModal.place.lat) < 0.0001 && Math.abs(p.lon - saveModal.place.lon) < 0.0001);
            if (existingCoords) {
                setSaveModal(p => ({ ...p, warning: `This place is already saved as: ${existingCoords.label}`, warningType: 'coords', existingId: existingCoords.id }));
                return;
            }
        }
    }

    StorageService.savePlace({
      ...saveModal.place,
      label,
      category: 'custom'
    });
    setSavedPlaces(StorageService.getSaved());
    setSaveModal({ open: false, mode: 'create', place: null, label: '', warning: '', warningType: '', existingId: null });
    
    // Update destination if we just edited the active destination
    if (destination && destination.id === saveModal.place.id) {
       setDestination(p => ({ ...p, label }));
    }
  };
  
  const handleDeleteSaved = (id) => {
    if (window.confirm("Delete this saved place?")) {
       StorageService.deleteSaved(id);
       setSavedPlaces(StorageService.getSaved());
       if (destination && destination.id === id) {
           setDestination(p => { const newP = {...p}; delete newP.id; delete newP.label; return newP; });
       }
    }
  };

  const handleMapClick = (latlng) => {
    if (navStateMachine === 'IDLE' || navStateMachine === 'ROUTE_READY') {
      calculateRoute(latlng.lat, latlng.lng, "Selected Map Location", "custom");
    }
  };
  
  const simulateOutage = () => {
    fetch('http://localhost:8000/api/sim/trigger_outage', { method: 'POST' }).catch(()=>{});
    setNavStateMachine('GNSS_OUTAGE');
  };
  
  const restoreGNSS = () => {
    fetch('http://localhost:8000/api/sim/recover_gnss', { method: 'POST' }).catch(()=>{});
    setNavStateMachine('NAVIGATING');
  };

  const centerPos = navState.lat !== 0 ? [navState.lat, navState.lon] : [28.6139, 77.2090];
  const isDR = navState.mode === 'DEAD_RECKONING';
  let currentIcon = isDR ? drIcon : navIcon;
  if (navState.mode === 'WAITING') currentIcon = statIcon;

  return (
    <div className="app-container">
      <div className="map-layer">
        <MapContainer center={centerPos} zoom={16} style={{ height: '100%', width: '100%' }}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={20} />
          <MapEvents onMapClick={handleMapClick} />
          
          {routeData && <Polyline positions={routeData.coordinates} color="#6366f1" weight={8} opacity={0.7} />}
          {fusedPath.length > 0 && <Polyline positions={fusedPath} color={isDR ? "#f59e0b" : "#3b82f6"} weight={6} opacity={0.8} />}
          {navState.lat !== 0 && <Marker position={[navState.lat, navState.lon]} icon={currentIcon} />}
          {destination && <Marker position={[destination.lat, destination.lon]} icon={destIcon} />}
          
          <MapController lat={centerPos[0]} lon={centerPos[1]} isTracking={isTracking} setIsTracking={setIsTracking} firstLockRef={firstLockRef} destLat={destination?.lat} destLon={destination?.lon} />
        </MapContainer>
      </div>

      <div className="ui-layer">
        
        {/* Top Search Bar */}
        <div className="search-container">
          <div className="panel search-input-box interactive-panel">
            <Search size={20} color="var(--text-secondary)" />
            <input 
              type="text" 
              placeholder="Search destination, building, place..." 
              value={searchQuery}
              onChange={(e) => { setSearchQuery(e.target.value); setShowSearch(true); }}
              onFocus={() => setShowSearch(true)}
            />
            {showSearch && <X size={20} cursor="pointer" onClick={() => { setShowSearch(false); setSearchQuery(''); }} />}
          </div>

          {showSearch && (
            <div className="panel search-dropdown interactive-panel">
              {searchResults.length > 0 ? (
                <>
                  <div className="search-section">Suggestions</div>
                  {searchResults.map(res => (
                    <div key={res.place_id} className="search-item" onClick={() => calculateRoute(parseFloat(res.lat), parseFloat(res.lon), res.display_name.split(',')[0], "search")}>
                      <div className="search-item-icon"><MapPin size={16} /></div>
                      <div>
                        <div style={{fontWeight: 600}}>{res.display_name.split(',')[0]}</div>
                        <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>{res.type} - {res.address?.city || ''}</div>
                      </div>
                    </div>
                  ))}
                </>
              ) : (
                <>
                  <div className="search-section">Saved Places</div>
                  <div className="saved-tags-container" style={{display: 'flex', gap: '10px', padding: '10px 16px', overflowX: 'auto', whiteSpace: 'nowrap'}}>
                    {savedPlaces.length > 0 ? savedPlaces.map(p => (
                      <button key={p.id} className="btn-outline" style={{padding: '8px 12px', fontSize: '0.9rem', flexShrink: 0}} onClick={() => {
                        calculateRoute(p.lat, p.lon, p.name, 'saved', p);
                      }}>
                        <Star size={16} style={{display:'inline', marginRight: '6px', color: 'var(--warning-amber)'}}/> {p.label}
                      </button>
                    )) : (
                      <div style={{fontSize: '0.85rem', color: 'var(--text-secondary)'}}>No saved places yet.</div>
                    )}
                  </div>

                  {recentPlaces.length > 0 && (
                    <>
                      <div className="search-section">Recent</div>
                      {recentPlaces.slice(0,5).map((p, i) => (
                        <div key={i} className="search-item" onClick={() => calculateRoute(p.lat, p.lon, p.name, "recent")}>
                          <div className="search-item-icon"><Clock size={16} /></div>
                          <div>{p.name}</div>
                        </div>
                      ))}
                    </>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        <div className="panels-container">
          <div className="nav-card panel interactive-panel">
            
            {navStateMachine === 'IDLE' && (
              <div style={{textAlign: 'center', padding: '20px 0'}}>
                <Map size={32} style={{margin: '0 auto 10px auto', opacity: 0.5}} />
                <div>Where to?</div>
              </div>
            )}
            
            {navStateMachine === 'ROUTE_CALCULATING' && (
              <div style={{textAlign: 'center', padding: '20px 0', fontWeight: 600}}>CALCULATING ROUTE...</div>
            )}
            
            {(navStateMachine === 'ROUTE_READY' || navStateMachine === 'NAVIGATING' || navStateMachine === 'GNSS_OUTAGE') && destination && routeData && (
              <>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px'}}>
                  <div>
                    {destination.label ? (
                        <>
                          <strong style={{fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '6px'}}>
                            <Star size={18} color="var(--warning-amber)" /> {destination.label}
                          </strong>
                          <div style={{color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '2px'}}>{destination.name}</div>
                        </>
                    ) : (
                        <strong style={{fontSize: '1.1rem'}}>{destination.name}</strong>
                    )}
                  </div>
                  {navStateMachine === 'ROUTE_READY' && (
                    <div style={{display: 'flex', gap: '8px'}}>
                      {destination.id ? (
                          <>
                            <button className="btn-outline" style={{padding: '6px 10px'}} onClick={() => openSaveModal(destination, 'edit')}>Edit</button>
                            <button className="btn-outline" style={{padding: '6px 10px', color: 'var(--alert-red)', borderColor: 'rgba(239, 68, 68, 0.3)'}} onClick={() => handleDeleteSaved(destination.id)}>Delete</button>
                          </>
                      ) : (
                          <button className="btn-outline" style={{padding: '6px 12px'}} onClick={() => openSaveModal(destination, 'create')}>Save Place</button>
                      )}
                    </div>
                  )}
                </div>
                
                <div className="route-metrics">
                  <div>
                    <div style={{fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-primary)'}}>{(routeData.distance / 1000).toFixed(1)} km</div>
                    <div style={{fontSize: '0.75rem', fontWeight: 600}}>DISTANCE</div>
                  </div>
                  <div style={{textAlign: 'right'}}>
                    <div style={{fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-primary)'}}>{Math.ceil(routeData.duration / 60)} min</div>
                    <div style={{fontSize: '0.75rem', fontWeight: 600}}>ETA</div>
                  </div>
                </div>

                <div style={{height: '1px', background: 'var(--panel-border)', margin: '16px 0'}}></div>
                
                {navStateMachine === 'ROUTE_READY' && (
                  <button className="btn-primary" style={{width: '100%'}} onClick={handleStartNav}>
                    <Play size={18} /> START NAVIGATION
                  </button>
                )}
                
                {navStateMachine === 'NAVIGATING' && (
                  <button className="btn-primary" style={{width: '100%', background: 'var(--warning-amber)', color: '#000'}} onClick={simulateOutage}>
                    <PowerOff size={18} /> SIMULATE GNSS OUTAGE
                  </button>
                )}
                
                {navStateMachine === 'GNSS_OUTAGE' && (
                  <div style={{background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--alert-red)', padding: '12px', borderRadius: '8px', textAlign: 'center'}}>
                    <div style={{color: 'var(--alert-red)', fontWeight: 'bold', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'}}>
                      <AlertTriangle size={18} /> GNSS OUTAGE
                    </div>
                    <div style={{fontSize: '0.85rem', marginBottom: '12px'}}>AI DEAD RECKONING ACTIVE</div>
                    <button className="btn-primary" style={{width: '100%', background: 'var(--ok-green)'}} onClick={restoreGNSS}>RESTORE GNSS</button>
                  </div>
                )}
              </>
            )}
            
            {navStateMachine === 'ARRIVED' && (
              <div style={{textAlign: 'center', padding: '20px 0'}}>
                <div style={{color: 'var(--ok-green)', fontSize: '1.5rem', fontWeight: 800, marginBottom: '10px'}}>YOU HAVE ARRIVED</div>
                <button className="btn-outline" onClick={() => { setNavStateMachine('IDLE'); setDestination(null); setRouteData(null); }}>Clear</button>
              </div>
            )}
          </div>

          <div className="map-controls interactive-panel">
            <button className="icon-btn" onClick={() => setShowSettings(!showSettings)}><Settings size={22} /></button>
            <button className={`icon-btn ${isTracking ? 'active' : ''}`} onClick={() => setIsTracking(true)}><Crosshair size={22} /></button>
          </div>
        </div>

        {/* --- Settings Modal --- */}
        {showSettings && (
          <div className="overlay-screen" onClick={() => setShowSettings(false)}>
            <div className="panel modal interactive-panel" onClick={e => e.stopPropagation()}>
              <h3 style={{margin: '0 0 20px 0', display: 'flex', justifyContent: 'space-between'}}>
                Settings <X size={20} cursor="pointer" onClick={() => setShowSettings(false)} />
              </h3>
              
              <div className="settings-row">
                <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}><Monitor size={18}/> Theme</div>
                <select value={theme} onChange={e => setTheme(e.target.value)} style={{padding: '6px', background: 'var(--input-bg)', color: 'var(--text-primary)', border: '1px solid var(--panel-border)', borderRadius: '6px'}}>
                  <option value="system">System</option>
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                </select>
              </div>

              <div className="settings-row">
                <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}><Volume2 size={18}/> Voice Nav</div>
                <input type="checkbox" checked={voiceEnabled} onChange={e => setVoiceEnabled(e.target.checked)} style={{transform: 'scale(1.5)'}} />
              </div>

              <div style={{marginTop: '20px', fontSize: '0.8rem', color: 'var(--text-secondary)', textAlign: 'center'}}>
                IDR Navigation - Phase 3 Prototype
              </div>
            </div>
          </div>
        )}

        {/* --- Custom Save Modal --- */}
        {saveModal.open && (
          <div className="overlay-screen" style={{zIndex: 10000}} onClick={() => setSaveModal({open:false, place: null, label: '', warning: ''})}>
            <div className="panel modal interactive-panel" onClick={e => e.stopPropagation()}>
              <h3 style={{margin: '0 0 16px 0'}}>{saveModal.mode === 'edit' ? 'Edit Place' : 'Save Place'}</h3>
              <div style={{marginBottom: '16px'}}>
                 <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>Address</div>
                 <div style={{fontWeight: 600, fontSize: '0.95rem', marginTop: '4px'}}>{saveModal.place.name}</div>
              </div>
              <div style={{marginBottom: '16px'}}>
                 <div style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>Give this place a name</div>
                 <input 
                    type="text" 
                    value={saveModal.label} 
                    onChange={e => setSaveModal(p => ({...p, label: e.target.value, warning: ''}))}
                    style={{width: '100%', padding: '12px', marginTop: '8px', background: 'var(--input-bg)', color: 'var(--text-primary)', border: '1px solid var(--panel-border)', borderRadius: '8px', fontSize: '1rem', outline: 'none'}}
                    placeholder="e.g. Home, College, Gym"
                    autoFocus
                 />
              </div>
              
              {saveModal.warning && (
                  <div style={{color: 'var(--warning-amber)', marginBottom: '16px', fontSize: '0.9rem', background: 'rgba(245, 158, 11, 0.1)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(245, 158, 11, 0.3)'}}>
                      <div style={{marginBottom: '8px'}}>{saveModal.warning}</div>
                      {saveModal.warningType === 'name' && (
                          <div style={{display: 'flex', gap: '8px'}}>
                              <button className="btn-outline" style={{padding: '6px 12px', fontSize: '0.8rem'}} onClick={() => handleConfirmSave(true)}>Replace Existing</button>
                          </div>
                      )}
                      {saveModal.warningType === 'coords' && (
                          <div style={{display: 'flex', gap: '8px'}}>
                              <button className="btn-outline" style={{padding: '6px 12px', fontSize: '0.8rem'}} onClick={() => {
                                  const existing = savedPlaces.find(p => p.id === saveModal.existingId);
                                  setSaveModal({open: false, place: null, label: '', warning: ''});
                                  calculateRoute(existing.lat, existing.lon, existing.name, 'saved', existing);
                              }}>Use Existing</button>
                              <button className="btn-outline" style={{padding: '6px 12px', fontSize: '0.8rem'}} onClick={() => handleConfirmSave(true)}>Save Anyway</button>
                          </div>
                      )}
                  </div>
              )}

              <div style={{display: 'flex', justifyContent: 'flex-end', gap: '12px'}}>
                 <button className="btn-outline" onClick={() => setSaveModal({open:false, place: null, label: '', warning: ''})}>CANCEL</button>
                 <button className="btn-primary" onClick={() => handleConfirmSave(false)}>SAVE</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
