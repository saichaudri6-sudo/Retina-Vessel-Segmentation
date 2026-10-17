/* RetinaVessel instrument controls — vanilla JS, no framework.
   1) slider readout labels
   2) plate viewport: scroll zoom, drag pan, keyboard, double-click fit
   3) number-key shortcuts for plate modes
   4) busy hairline surge while a fragment computation is in flight      */
(function () {
  "use strict";

  /* --------------------------------------------------- slider readouts */
  var FMT = {
    f1: function (v) { return v.toFixed(1); },
    f2: function (v) { return v.toFixed(2); },
    pct: function (v) { return Math.round(v * 100) + "%"; }
  };

  function initTuneReadouts() {
    var form = document.getElementById("tuneform");
    if (!form || form.dataset.bound) return;
    form.dataset.bound = "1";
    var sliders = form.querySelectorAll("input[type=range]");
    Array.prototype.forEach.call(sliders, function (r) {
      var block = r.closest(".ctl");
      var out = block ? block.querySelector("b[data-fmt]") : null;
      if (!out) return;
      var fmt = FMT[out.dataset.fmt];
      if (!fmt) return;
      out.textContent = fmt(parseFloat(r.value));
      r.addEventListener("input", function () {
        out.textContent = fmt(parseFloat(r.value));
      });
    });
  }

  /* ------------------------------------------------------- plate viewer */
  var MIN_FACTOR = 0.25, MAX_FACTOR = 8;

  function initViewer() {
    var vp = document.getElementById("viewport");
    var img = document.getElementById("viewport-img");
    var hud = document.getElementById("zoomhud");
    if (!vp || !img || !hud || img.dataset.viewerBound) return;
    img.dataset.viewerBound = "1";

    var tx = 0, ty = 0, scale = 1, fit = 1;

    function clampPan() {
      var vw = vp.clientWidth, vh = vp.clientHeight;
      var iw = img.naturalWidth * scale, ih = img.naturalHeight * scale;
      var mx = Math.max(0, (iw - vw) / 2 + 24), my = Math.max(0, (ih - vh) / 2 + 24);
      tx = Math.min(mx, Math.max(-mx, tx));
      ty = Math.min(my, Math.max(-my, ty));
    }

    function apply() {
      clampPan();
      img.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
      hud.textContent = Math.round(scale * 100) + "%";
    }

    function fitImage() {
      var vw = vp.clientWidth || 1, vh = vp.clientHeight || 1;
      var iw = img.naturalWidth || 1, ih = img.naturalHeight || 1;
      fit = Math.min(vw / iw, vh / ih);
      scale = fit; tx = 0; ty = 0; apply();
    }

    function zoom(factor) {
      var next = scale * factor;
      scale = Math.min(Math.max(fit * MIN_FACTOR, next), fit * MAX_FACTOR);
      apply();
      vp.classList.add("touched");
    }

    function whenLoaded(fn) {
      if (img.naturalWidth > 0) fn();
      else img.addEventListener("load", fn, { once: true });
    }

    whenLoaded(fitImage);

    window.addEventListener("resize", function () {
      whenLoaded(function () { if (!vp.classList.contains("touched")) fitImage(); else apply(); });
    });

    vp.addEventListener("wheel", function (e) {
      e.preventDefault();
      zoom(Math.exp(-e.deltaY * 0.0016));
    }, { passive: false });

    var dragging = false, sx = 0, sy = 0, stx = 0, sty = 0;
    vp.addEventListener("pointerdown", function (e) {
      dragging = true; sx = e.clientX; sy = e.clientY; stx = tx; sty = ty;
      vp.classList.add("dragging", "touched");
      vp.setPointerCapture(e.pointerId);
    });
    vp.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      tx = stx + (e.clientX - sx);
      ty = sty + (e.clientY - sy);
      apply();
    });
    function release() { dragging = false; vp.classList.remove("dragging"); }
    vp.addEventListener("pointerup", release);
    vp.addEventListener("pointercancel", release);

    vp.addEventListener("dblclick", fitImage);

    vp.addEventListener("keydown", function (e) {
      var step = 40;
      if (e.key === "+" || e.key === "=") { zoom(1.25); e.preventDefault(); e.stopPropagation(); }
      else if (e.key === "-" || e.key === "_") { zoom(0.8); e.preventDefault(); e.stopPropagation(); }
      else if (e.key === "0") { fitImage(); e.preventDefault(); e.stopPropagation(); }
      else if (e.key === "ArrowLeft") { tx -= step; apply(); e.preventDefault(); e.stopPropagation(); }
      else if (e.key === "ArrowRight") { tx += step; apply(); e.preventDefault(); e.stopPropagation(); }
      else if (e.key === "ArrowUp") { ty -= step; apply(); e.preventDefault(); e.stopPropagation(); }
      else if (e.key === "ArrowDown") { ty += step; apply(); e.preventDefault(); e.stopPropagation(); }
    });
  }

  /* --------------------------------------------------- keyboard: modes */
  document.addEventListener("keydown", function (e) {
    var tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (/^[1-7]$/.test(e.key)) {
      var chip = document.querySelector(".chip[data-key='" + e.key + "']");
      if (chip) chip.click();
      return;
    }
  });

  /* ------------------------------------------------ htmx lifecycle hooks */
  function bindLifecycle() {
    var hair = document.getElementById("busyhair");
    document.body.addEventListener("htmx:beforeRequest", function () {
      if (hair) hair.classList.add("on");
    });
    document.body.addEventListener("htmx:afterRequest", function () {
      if (hair) hair.classList.remove("on");
    });
    document.body.addEventListener("htmx:afterSwap", function (e) {
      if (e.target && (e.target.id === "bay" || e.target.id === "eyepiece")) initViewer();
    });
  }

  function boot() {
    initTuneReadouts();
    initViewer();
    bindLifecycle();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
