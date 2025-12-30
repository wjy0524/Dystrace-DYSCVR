window.resetWebGazer = function () {
    console.log("resetWebGazer called ✅");
  
    if (window.webgazer) {
      // 1️⃣ Hide any on‑screen WebGazer UI
      webgazer
        .showVideo(false)
        .showFaceOverlay(false)
        .showPredictionPoints(false);
  
      // 2️⃣ Stop and clear the gaze listener + pause the tracker
      webgazer.clearGazeListener?.();
      webgazer.pause?.();
  
      // 3️⃣ Clear stored gaze data
      webgazer.clearData?.();
    }
  
    // 4️⃣ Remove **all** canvases (prediction overlays)
    document.querySelectorAll('canvas').forEach(el => el.remove());
  
    // 5️⃣ Remove any WebGazer DOM elements or your own indicator
    document.querySelectorAll(
      '#webgazerVideoFeed, #webgazerFaceOverlay, #webgazerFaceFeedbackBox,' +
      '.webgazerDot, #gazeIndicator'
    ).forEach(el => el.remove());
  
    // 6️⃣ Clear your global reference
    window._gazeIndicator = null;
  };