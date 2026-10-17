/* ============================================================
   ColorBends — vanilla WebGL port of ReactBits' ColorBends
   (DPI-capped, paused when hidden, static under reduced motion,
   graceful fallback to the CSS room gradients without WebGL)
   Params follow the studio URL: rotation=95, colors #000/#000/4f4f4f,
   speed 0.2, noise 0.15, intensity 1.5, bandWidth 6, warp 1, freq 1.
   ============================================================ */
(() => {
  const canvas = document.getElementById("bends");
  if (!canvas) return;  const gl = canvas.getContext("webgl", {
    antialias: false,
    powerPreference: "high-performance",
    alpha: true,
    premultipliedAlpha: true
  });
  if (!gl) { canvas.remove(); return; } // fallback: body::before room gradients

  const MAX_COLORS = 8;

  const frag = `
precision highp float;
#define MAX_COLORS ${MAX_COLORS}
uniform vec2 uCanvas;
uniform float uTime;
uniform float uSpeed;
uniform vec2 uRot;
uniform int uColorCount;
uniform vec3 uColors[MAX_COLORS];
uniform int uTransparent;
uniform float uScale;
uniform float uFrequency;
uniform float uWarpStrength;
uniform vec2 uPointer;
uniform float uMouseInfluence;
uniform float uParallax;
uniform float uNoise;
uniform int uIterations;
uniform float uIntensity;
uniform float uBandWidth;
varying vec2 vUv;

void main() {
  float t = uTime * uSpeed;
  vec2 p = vUv * 2.0 - 1.0;
  p += uPointer * uParallax * 0.1;
  vec2 rp = vec2(p.x * uRot.x - p.y * uRot.y, p.x * uRot.y + p.y * uRot.x);
  vec2 q = vec2(rp.x * (uCanvas.x / uCanvas.y), rp.y);
  q /= max(uScale, 0.0001);
  q /= 0.5 + 0.2 * dot(q, q);
  q += 0.2 * cos(t) - 7.56;
  vec2 toward = (uPointer - rp);
  q += toward * uMouseInfluence * 0.2;

  for (int j = 0; j < 5; j++) {
    if (j >= uIterations - 1) break;
    vec2 rr = sin(1.5 * (q.yx * uFrequency) + 2.0 * cos(q * uFrequency));
    q += (rr - q) * 0.15;
  }

  vec3 col = vec3(0.0);
  float a = 1.0;

  if (uColorCount > 0) {
    vec2 s = q;
    vec3 sumCol = vec3(0.0);
    float cover = 0.0;
    for (int i = 0; i < MAX_COLORS; ++i) {
      if (i >= uColorCount) break;
      s -= 0.01;
      vec2 r = sin(1.5 * (s.yx * uFrequency) + 2.0 * cos(s * uFrequency));
      float m0 = length(r + sin(5.0 * r.y * uFrequency - 3.0 * t + float(i)) / 4.0);
      float kBelow = clamp(uWarpStrength, 0.0, 1.0);
      float kMix = pow(kBelow, 0.3);
      float gain = 1.0 + max(uWarpStrength - 1.0, 0.0);
      vec2 disp = (r - s) * kBelow;
      vec2 warped = s + disp * gain;
      float m1 = length(warped + sin(5.0 * warped.y * uFrequency - 3.0 * t + float(i)) / 4.0);
      float m = mix(m0, m1, kMix);
      float w = 1.0 - exp(-uBandWidth / exp(uBandWidth * m));
      sumCol += uColors[i] * w;
      cover = max(cover, w);
    }
    col = clamp(sumCol, 0.0, 1.0);
    a = uTransparent > 0 ? cover : 1.0;
  }

  col *= uIntensity;

  if (uNoise > 0.0001) {
    float n = fract(sin(dot(gl_FragCoord.xy + vec2(uTime), vec2(12.9898, 78.233))) * 43758.5453123);
    col += (n - 0.5) * uNoise;
    col = clamp(col, 0.0, 1.0);
  }

  vec3 rgb = (uTransparent > 0) ? col * a : col;
  gl_FragColor = vec4(rgb, a);
}`;

  const vert = `
precision highp float;
attribute vec2 aPos;
varying vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("colorbends shader:", gl.getShaderInfoLog(s));
      return null;
    }
    return s;
  };
  const vs = sh(gl.VERTEX_SHADER, vert), fs = sh(gl.FRAGMENT_SHADER, frag);
  if (!vs || !fs) { canvas.remove(); return; }
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error("colorbends link:", gl.getProgramInfoLog(prog));
    canvas.remove();
    return;
  }
  gl.useProgram(prog);

  // fullscreen triangle
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "aPos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const u = name => gl.getUniformLocation(prog, name);
  const uCanvas = u("uCanvas"), uTime = u("uTime"), uSpeed = u("uSpeed"),
    uRot = u("uRot"), uColorCount = u("uColorCount"), uColors = u("uColors[0]"),
    uTransparent = u("uTransparent"), uScale = u("uScale"), uFrequency = u("uFrequency"),
    uWarpStrength = u("uWarpStrength"), uPointer = u("uPointer"),
    uMouseInfluence = u("uMouseInfluence"), uParallax = u("uParallax"),
    uNoise = u("uNoise"), uIterations = u("uIterations"),
    uIntensity = u("uIntensity"), uBandWidth = u("uBandWidth");

  const toVec3 = hex => {
    const h = hex.replace("#", "").trim();
    const p = n => parseInt(n, 16) / 255;
    if (h.length === 3) return [p(h[0] + h[0]), p(h[1] + h[1]), p(h[2] + h[2])];
    return [p(h.slice(0, 2)), p(h.slice(2, 4)), p(h.slice(4, 6))];
  };
  const colors = ["000000", "000000", "4f4f4f"].map(toVec3);
  const colorArr = new Float32Array(MAX_COLORS * 3);
  colors.forEach((c, i) => colorArr.set(c, i * 3));
  gl.uniform3fv(uColors, colorArr);
  gl.uniform1i(uColorCount, colors.length);
  gl.uniform1i(uTransparent, 0);
  gl.uniform2f(uCanvas, 1, 1);
  gl.uniform1f(uSpeed, 0.2);
  gl.uniform1f(uScale, 1.35);
  gl.uniform1f(uFrequency, 1.35);
  gl.uniform1f(uWarpStrength, 1);
  gl.uniform1f(uMouseInfluence, 1);
  gl.uniform1f(uParallax, 0.5);
  gl.uniform1f(uNoise, 0.15);
  gl.uniform1i(uIterations, 1);
  gl.uniform1f(uIntensity, 2.2);
  gl.uniform1f(uBandWidth, 6);

  // rotation = 95deg, no auto-rotate
  const rad = (95 * Math.PI) / 180;
  gl.uniform2f(uRot, Math.cos(rad), Math.sin(rad));

  const pointer = { tx: 0, ty: 0, x: 0, y: 0 };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  window.addEventListener("pointermove", e => {
    pointer.tx = (e.clientX / innerWidth) * 2 - 1;
    pointer.ty = -((e.clientY / innerHeight) * 2 - 1);
  }, { passive: true });

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(innerWidth));
    const h = Math.max(1, Math.round(innerHeight));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform2f(uCanvas, w, h);
  };
  window.addEventListener("resize", resize);
  resize();

  const start = performance.now();
  const draw = now => {
    const dt = (now - (draw.last || now)) / 1000;
    draw.last = now;
    const px = pointer.x + (pointer.tx - pointer.x) * Math.min(1, dt * 8);
    const py = pointer.y + (pointer.ty - pointer.y) * Math.min(1, dt * 8);
    pointer.x = px; pointer.y = py;
    gl.uniform2f(uPointer, px, py);
    gl.uniform1f(uTime, (now - start) / 1000);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };

  if (reduced) {
    gl.uniform2f(uPointer, 0, 0);
    gl.uniform1f(uTime, 12.5); // one settled frame
    draw(performance.now());
  } else {
    // timer-driven (not rAF) so one committed warped frame lands even in
    // throttled/pictureless environments; background tabs settle at ~1 fps
    // via native timer clamping, which is enough for speed 0.2 bends
    draw(performance.now());
    const loop = () => {
      draw(performance.now());
      setTimeout(loop, 33);
    };
    setTimeout(loop, 0);
  }
})();
