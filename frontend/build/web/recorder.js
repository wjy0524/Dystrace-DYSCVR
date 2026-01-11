let mediaRecorder;
let chunks = [];

// Start recording (returns a Promise)
async function startMicRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  chunks = [];
  mediaRecorder.ondataavailable = e => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  mediaRecorder.start();
}

// Stop recording and resolve with base64 string
function stopMicRecording() {
  return new Promise((resolve, reject) => {
    if (!mediaRecorder) {
      reject("Recording has not been started.");
      return;
    }
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/webm; codecs=opus" });
      const reader = new FileReader();
      reader.onloadend = () => {
        // Remove the "data:...;base64," prefix and keep only the base64 string
        const base64 = reader.result.split(",")[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    };
    mediaRecorder.stop();
  });
}

// Expose functions globally so Flutter can call them via JS interop
window.startMicRecording = startMicRecording;
window.stopMicRecording  = stopMicRecording;

