export const haversine = (lat1, lon1, lat2, lon2) => {
  const R = 6371e3;
  const p1 = lat1 * Math.PI/180;
  const p2 = lat2 * Math.PI/180;
  const dp = (lat2-lat1) * Math.PI/180;
  const dl = (lon2-lon1) * Math.PI/180;
  const a = Math.sin(dp/2) * Math.sin(dp/2) + Math.cos(p1) * Math.cos(p2) * Math.sin(dl/2) * Math.sin(dl/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c; // meters
};

export const parseOSRMInstruction = (step) => {
  if (!step || !step.maneuver) return "Continue straight";
  const { type, modifier } = step.maneuver;
  const name = step.name ? ` onto ${step.name}` : "";
  
  if (type === 'depart') return `Head ${modifier || 'straight'}${name}`;
  if (type === 'arrive') return `You will arrive at your destination`;
  
  let action = "Continue";
  if (type === 'turn') action = "Turn";
  if (type === 'roundabout') action = "Take the roundabout";
  if (type === 'merge') action = "Merge";
  if (type === 'on ramp') action = "Take the ramp";
  if (type === 'off ramp') action = "Take the exit";
  if (type === 'fork') action = "Keep";
  if (type === 'end of road') action = "At the end of the road, turn";
  
  const modStr = modifier ? ` ${modifier.replace('-', ' ')}` : "";
  return `${action}${modStr}${name}`;
};

// Generates voice string based on distance
export const generateVoiceAnnouncement = (instruction, distanceToTurn) => {
  if (distanceToTurn > 300) {
    return `In ${Math.round(distanceToTurn/100)*100} meters, ${instruction}`;
  } else if (distanceToTurn > 100) {
    return `In ${Math.round(distanceToTurn/50)*50} meters, ${instruction}`;
  } else {
    return instruction;
  }
};
