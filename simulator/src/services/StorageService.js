const StorageService = {
  get(key, defaultValue) {
    try {
      const val = localStorage.getItem(key);
      return val ? JSON.parse(val) : defaultValue;
    } catch {
      return defaultValue;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
      console.error('Storage full or unavailable');
    }
  },
  getRecent() { return this.get('recentPlaces', []); },
  addRecent(place) {
    let recents = this.getRecent().filter(p => p.name !== place.name && p.lat !== place.lat);
    recents.unshift({ ...place, lastUsedAt: Date.now() });
    if (recents.length > 10) recents = recents.slice(0, 10);
    this.set('recentPlaces', recents);
  },
  clearRecent() { this.set('recentPlaces', []); },
  
  getSaved() { return this.get('savedPlaces', []); },
  savePlace(place) {
    let saved = this.getSaved();
    if (place.id) {
       saved = saved.filter(p => p.id !== place.id);
    }
    saved.push({ ...place, id: place.id || Date.now().toString(), createdAt: place.createdAt || Date.now(), updatedAt: Date.now() });
    
    // Sort so most recently updated/used is first
    saved.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
    
    this.set('savedPlaces', saved);
  },
  deleteSaved(id) {
    this.set('savedPlaces', this.getSaved().filter(p => p.id !== id));
  },
  updateLastUsed(id) {
    let saved = this.getSaved();
    let place = saved.find(p => p.id === id);
    if (place) {
        place.updatedAt = Date.now();
        this.savePlace(place);
    }
  }
};
export default StorageService;
