/**
 * ═══════════════════════════════════════════════════════════════════════════
 * ISRO MISSION CONTROL · CONTROLLER (AEROSPACE GLASSMORPHIC EDITION)
 * High-performance 120Hz/60Hz GPU-accelerated navigation and live IST telemetry clock.
 * ═══════════════════════════════════════════════════════════════════════════
 */

(function() {
  console.log("🛰️ [ISRO Mission Control] Controller initializing...");

  // 1. Live IST Mission Clock
  function updateISTClock() {
    try {
      const now = new Date();
      const istString = now.toLocaleTimeString('en-US', { timeZone: 'Asia/Kolkata', hour12: false });
      const clockElem = document.getElementById('live-ist-time');
      if (clockElem) {
        clockElem.innerText = 'IST ' + istString;
      }
    } catch(e) {}
  }
  setInterval(updateISTClock, 1000);
  updateISTClock();

  // 2. Smooth ProMotion / 120fps requestAnimationFrame Smooth Glider
  let activeScrollAnim = null;

  window.smoothGlideTo = function(targetElement, offset = 12, duration = 850) {
    if (!targetElement) return;

    if (activeScrollAnim) {
      cancelAnimationFrame(activeScrollAnim);
      activeScrollAnim = null;
    }

    const scroller = document.querySelector('section.main') ||
                     document.querySelector('[data-testid="stMain"]') ||
                     document.documentElement;

    const isWindow = (scroller === document.documentElement || scroller === document.body);
    const startY = isWindow ? (window.pageYOffset || document.documentElement.scrollTop) : scroller.scrollTop;

    const targetRect = targetElement.getBoundingClientRect();
    const scrollerRect = isWindow ? { top: 0 } : scroller.getBoundingClientRect();
    const targetY = startY + (targetRect.top - scrollerRect.top) - offset;
    const distance = targetY - startY;

    if (Math.abs(distance) < 4) return;

    const startTime = performance.now();

    function scrollStep(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Smooth easeInOutCubic curve
      const ease = progress < 0.5
        ? 4 * progress * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 3) / 2;

      const currentY = startY + (distance * ease);

      if (isWindow) {
        window.scrollTo(0, currentY);
      } else {
        scroller.scrollTop = currentY;
      }

      if (progress < 1) {
        activeScrollAnim = requestAnimationFrame(scrollStep);
      } else {
        activeScrollAnim = null;
      }
    }

    activeScrollAnim = requestAnimationFrame(scrollStep);
  };

  // Helper: Scroll to Pipeline Registration smoothly
  window.scrollToPipelineRegistration = function() {
    const target = document.getElementById('pipeline-registration-section') ||
                   document.querySelector('[data-testid="stTabs"]') ||
                   document.querySelector('.stTabs');
    if (target) {
      try {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch(e) {}
      window.smoothGlideTo(target, 12, 700);
    }
  };

  // Helper: Scroll back to top
  window.scrollToCosmicGateway = function() {
    const target = document.getElementById('isro-cosmic-gateway') ||
                   document.querySelector('.block-container') ||
                   document.body;
    if (target) {
      try {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch(e) {}
      try { window.scrollTo({ top: 0, behavior: 'smooth' }); } catch(e) {}
    }
  };

  // 3. Global Event Delegation for floating buttons
  document.addEventListener('click', function(e) {
    const target = e.target;
    if (!target) return;

    // Never intercept Streamlit native widgets
    if (target.closest('.stButton') ||
        target.closest('[data-testid^="stBaseButton"]') ||
        target.closest('button[kind]') ||
        target.closest('.stSelectbox') ||
        target.closest('.stTextInput') ||
        target.closest('.stDownloadButton')) {
      return;
    }

    if (target.closest('[data-action="jump-pipeline"]') || target.closest('.floating-hud-btn[title*="Pipeline"]')) {
      e.preventDefault();
      window.scrollToPipelineRegistration();
      return;
    }
    if (target.closest('[data-action="jump-gateway"]') || target.closest('.floating-hud-btn[title*="Gateway"]')) {
      e.preventDefault();
      window.scrollToCosmicGateway();
      return;
    }
  }, false);

  console.log("✅ [ISRO Mission Control] Controller active.");
})();
