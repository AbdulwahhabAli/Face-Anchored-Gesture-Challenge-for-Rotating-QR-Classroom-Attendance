// Selfie camera helper.
// Live preview via getUserMedia when the page is a secure context (HTTPS or localhost);
// otherwise falls back to <input type=file capture=user>, which opens the front camera on phones.
const SelfieCamera = {
  stream: null,
  live: false,

  async start(video) {
    this.live = false;
    if (!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) return false;
    try {
      if (!this.stream) {
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: false, video: { facingMode: "user", width: { ideal: 1920 }, height: { ideal: 1440 } } });
      }
      video.srcObject = this.stream;
      await video.play();
      this.live = true;
    } catch (e) {
      console.warn("camera unavailable, using file capture", e);
    }
    return this.live;
  },

  // Grab the current frame (un-mirrored) as a JPEG blob.
  capture(video, quality = 0.92) {
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    return new Promise(res => c.toBlob(res, "image/jpeg", quality));
  },

  stop() {
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.stream = null; this.live = false;
  },
};
