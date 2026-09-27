class VoiceEngine {
  constructor() {
    this.enabled = true;
    this.volume = 1.0;
    this.rate = 1.0;
    this.synth = window.speechSynthesis;
  }

  setEnabled(val) { this.enabled = val; }
  setVolume(val) { this.volume = val; }
  setRate(val) { this.rate = val; }

  speak(text) {
    if (!this.enabled || !this.synth) return;
    this.synth.cancel(); // Stop current speech immediately
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.volume = this.volume;
    utterance.rate = this.rate;
    // Attempt to use a clearer voice if available
    const voices = this.synth.getVoices();
    const googleVoice = voices.find(v => v.name.includes('Google'));
    if (googleVoice) utterance.voice = googleVoice;
    this.synth.speak(utterance);
  }

  stop() {
    if (this.synth) this.synth.cancel();
  }
}

const VoiceService = new VoiceEngine();
export default VoiceService;
