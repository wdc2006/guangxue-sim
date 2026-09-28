# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, request, Response, jsonify

app = Flask(__name__)

# ============================================================
# 模块一：双光栅信号仿真器
# ============================================================
class GratingSimulator:
    def __init__(self):
        self.x = 0.0
        self.t = 0.0
        self.d = 1.0e-6          # 光栅周期 (m)
        self.N = 4               # 光学细分倍数
        self.theta = 0.02        # 莫尔夹角 (rad)
        self.mode = 'interference'
        self.dc = np.array([0.0, 0.0])
        self.gain_err = 1.0
        self.ortho_err = 0.0
        self.noise = 0.0

    def reset(self):
        self.x = 0.0
        self.t = 0.0

    @property
    def sensitivity(self):
        if self.mode == 'interference':
            return 2.0 * np.pi * self.N / self.d
        return 2.0 * np.pi / self.d

    @property
    def scale(self):
        if self.mode == 'interference':
            return self.d / (2.0 * np.pi * self.N)
        return self.d / (2.0 * np.pi)

    def advance(self, v, dt):
        self.x += v * dt
        self.t += dt

    def sample(self):
        phi = self.sensitivity * self.x
        I = np.cos(phi) + self.dc[0]
        Q = self.gain_err * np.sin(phi + self.ortho_err) + self.dc[1]
        if self.noise > 0:
            I += np.random.randn() * self.noise
            Q += np.random.randn() * self.noise
        return I, Q, phi

# ============================================================
# 模块二：相位解码器（椭圆校正 + atan2 + 解包裹）
# ============================================================
class PhaseDecoder:
    CALIB_N = 600

    def __init__(self):
        self.reset()

    def reset(self):
        self.offset = np.zeros(2)
        self.M = np.eye(2)
        self.buf = []
        self.calibrated = False
        self.phi_prev = None
        self.phi_unwrap = 0.0

    def collect(self, I, Q):
        if self.calibrated:
            return False
        self.buf.append((I, Q))
        if len(self.buf) >= self.CALIB_N:
            self._fit()
            return True
        return False

    def _fit(self):
        data = np.array(self.buf)
        self.offset = data.mean(axis=0)
        d = data - self.offset
        C = (d.T @ d) / len(d)

        G = np.sqrt(max(2.0 * C[1, 1], 1e-12))
        se = np.clip(2.0 * C[0, 1] / G, -0.999, 0.999)
        ce = np.sqrt(1.0 - se * se)

        self.M = np.array([
            [1.0, 0.0],
            [-se / ce, 1.0 / (G * ce)]
        ])
        self.calibrated = True
        self.phi_prev = None

    def decode(self, I, Q):
        v = np.array([I, Q]) - self.offset
        c, s = self.M @ v
        phi = np.arctan2(s, c)

        if self.phi_prev is None:
            self.phi_unwrap = phi
        else:
            dphi = phi - self.phi_prev
            dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
            self.phi_unwrap += dphi

        self.phi_prev = phi
        return phi, self.phi_unwrap

# ============================================================
# 模块三：卡尔曼滤波
# ============================================================
class KalmanTracker:
    def __init__(self, dt=1e-4):
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.x = np.zeros(2)
        self.P = np.diag([1e-10, 1e-8])
        self.Q = np.diag([1e-20, 1e-16])
        self.R = np.array([[1e-18]])
        self.I2 = np.eye(2)

    def update(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - float(self.H @ self.x)
        S = float(self.H @ self.P @ self.H.T) + float(self.R)
        K = (self.P @ self.H.T) / S
        self.x = self.x + K.flatten() * y
        self.P = (self.I2 - K @ self.H) @ self.P
        return float(self.x[0])

# ============================================================
# 全局状态与 API
# ============================================================
sim = GratingSimulator()
decoder = PhaseDecoder()
kf = KalmanTracker(dt=1e-4)
N_PTS = 200
DT = 1e-4

@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global sim, decoder, kf
    sim.reset()
    decoder.reset()
    kf = KalmanTracker(dt=DT)
    return jsonify({'ok': True})

@app.route('/api/step', methods=['POST'])
def api_step():
    global sim, decoder, kf
    p = request.get_json(force=True)
    sim.d = float(p.get('d', 1.0)) * 1e-6
    sim.N = int(p.get('N', 4))
    sim.theta = float(p.get('theta', 20.0)) * 1e-3
    sim.mode = p.get('mode', 'interference')
    sim.noise = float(p.get('noise', 0.01))
    sim.dc = np.array([float(p.get('dcI', 0.05)), float(p.get('dcQ', -0.03))])
    sim.gain_err = float(p.get('gain', 1.05))
    sim.ortho_err = np.deg2rad(float(p.get('ortho', 3.0)))
    v = float(p.get('velocity', 20.0)) * 1e-6

    out = {'t': [], 'I': [], 'Q': [], 'pr': [], 'pu': [], 'xt': [], 'xm': []}
    for _ in range(N_PTS):
        sim.advance(v, DT)
        I, Q, _ = sim.sample()
        just = decoder.collect(I, Q)
        if just:
            decoder.phi_prev = None
        phi_raw, phi_unwrap = decoder.decode(I, Q)
        x_meas = phi_unwrap * sim.scale
        x_filt = kf.update(x_meas)

        out['t'].append(sim.t)
        out['I'].append(float(I))
        out['Q'].append(float(Q))
        out['pr'].append(float(phi_raw))
        out['pu'].append(float(phi_unwrap))
        out['xt'].append(float(sim.x))
        out['xm'].append(float(x_filt))

    out['calibrated'] = decoder.calibrated
    out['calibProgress'] = min(1.0, len(decoder.buf) / PhaseDecoder.CALIB_N)
    out['scale'] = sim.scale
    out['sensitivity'] = sim.sensitivity
    out['moireMag'] = 1.0 / max(sim.theta, 1e-6)
    return jsonify(out)

# ============================================================
# 前端页面 (HTML + JS)
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>双光栅位移测量仿真</title>
<style>
:root{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2128; --border:#30363d;
  --fg:#c9d1d9; --muted:#8b949e;
  --blue:#58a6ff; --green:#3fb950; --orange:#d29922;
  --red:#f85149; --purple:#bc8cff; --cyan:#39d0d8;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0; background:var(--bg); color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  font-size:13px; line-height:1.45; overflow:hidden;
}
header{height:54px; display:flex; align-items:center; gap:14px; padding:0 18px; background:var(--panel); border-bottom:1px solid var(--border);}
header h1{font-size:15px; margin:0; font-weight:600; letter-spacing:.3px}
header .sub{font-size:11px; color:var(--muted)}
.logo{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,#1f6feb,#3fb950);display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;color:#fff;}
.spacer{flex:1}
button{background:var(--panel2); color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:6px 14px; font-size:12px; cursor:pointer; transition:.15s;}
button:hover{background:#22272e;border-color:#444c56}
button.primary{background:#238636;border-color:#2ea043;color:#fff}
button.primary:hover{background:#2ea043}
button.primary.on{background:#da3633;border-color:#f85149}
.status{font-size:11px;padding:4px 10px;border-radius:12px;background:var(--panel2);border:1px solid var(--border);color:var(--muted);}
.status.on{color:var(--green);border-color:#2ea043}
.layout{display:flex; height:calc(100% - 54px)}
aside{width:250px; flex-shrink:0; background:var(--panel); border-right:1px solid var(--border); overflow-y:auto; padding:12px;}
aside h2{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin:14px 0 8px;font-weight:600;}
aside h2:first-child{margin-top:0}
.field{margin-bottom:11px}
.field label{display:flex;justify-content:space-between;align-items:center;font-size:11.5px;color:var(--muted);margin-bottom:4px;}
.field .val{color:var(--blue);font-family:monospace;font-size:11.5px}
input[type=range]{width:100%;height:4px;-webkit-appearance:none;appearance:none;background:#21262d;border-radius:2px;outline:none;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:var(--blue);cursor:pointer;border:2px solid #0d1117;}
input[type=range]::-moz-range-thumb{width:13px;height:13px;border-radius:50%;background:var(--blue);cursor:pointer;border:2px solid #0d1117;}
select{width:100%;background:var(--panel2);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-size:12px;outline:none;}
main{flex:1;display:flex;flex-direction:column;padding:12px;gap:10px;min-width:0}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
.stat{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px 10px;}
.stat .k{font-size:10.5px;color:var(--muted);margin-bottom:2px}
.stat .v{font-size:15px;font-family:monospace;font-weight:600;color:var(--fg)}
.stat .u{font-size:10px;color:var(--muted);margin-left:2px}
.charts{flex:1;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:10px;min-height:0;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;display:flex;flex-direction:column;min-height:0;overflow:hidden;}
.card-h{font-size:11.5px;color:var(--muted);padding:7px 12px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;}
.card-h .legend{display:flex;gap:10px;font-size:10.5px}
.card-h .legend i{display:inline-block;width:8px;height:2px;border-radius:1px;margin-right:4px;vertical-align:middle;}
canvas{flex:1;width:100%;min-height:0;display:block}
</style>
</head>
<body>
<header>
  <div class="logo">◈</div>
  <div>
    <h1>双光栅位移测量仿真</h1>
    <div class="sub">Grating Interferometry · Phase Decoding · Sub-nm Resolution</div>
  </div>
  <div class="spacer"></div>
  <div class="status" id="status">已停止</div>
  <button id="btn-run" class="primary">▶ 开始</button>
  <button id="btn-reset">↺ 重置</button>
</header>
<div class="layout">
  <aside>
    <h2>光栅与光路</h2>
    <div class="field">
      <label>工作模式</label>
      <select id="p-mode">
        <option value="interference">双光栅干涉（光学细分）</option>
        <option value="moire">莫尔条纹（空间放大）</option>
      </select>
    </div>
    <div class="field">
      <label>光栅周期 d <span class="val" id="v-d">1.00 μm</span></label>
      <input type="range" id="p-d" min="0.5" max="5" step="0.1" value="1.0">
    </div>
    <div class="field">
      <label>光学细分倍数 N <span class="val" id="v-N">4 ×</span></label>
      <input type="range" id="p-N" min="1" max="32" step="1" value="4">
    </div>
    <div class="field">
      <label>莫尔夹角 θ <span class="val" id="v-theta">20 mrad</span></label>
      <input type="range" id="p-theta" min="2" max="100" step="1" value="20">
    </div>
    <h2>运动与噪声</h2>
    <div class="field">
      <label>运动速度 v <span class="val" id="v-v">20 μm/s</span></label>
      <input type="range" id="p-v" min="1" max="200" step="1" value="20">
    </div>
    <div class="field">
      <label>高斯噪声 σ <span class="val" id="v-noise">0.010</span></label>
      <input type="range" id="p-noise" min="0" max="0.1" step="0.002" value="0.01">
    </div>
    <h2>非理想误差</h2>
    <div class="field">
      <label>I 路直流偏置 <span class="val" id="v-dci">0.05</span></label>
      <input type="range" id="p-dci" min="-0.3" max="0.3" step="0.01" value="0.05">
    </div>
    <div class="field">
      <label>Q 路直流偏置 <span class="val" id="v-dcq">-0.03</span></label>
      <input type="range" id="p-dcq" min="-0.3" max="0.3" step="0.01" value="-0.03">
    </div>
    <div class="field">
      <label>增益失配 G <span class="val" id="v-gain">1.05</span></label>
      <input type="range" id="p-gain" min="0.7" max="1.4" step="0.01" value="1.05">
    </div>
    <div class="field">
      <label>正交相位误差 ε <span class="val" id="v-ortho">3.0°</span></label>
      <input type="range" id="p-ortho" min="-15" max="15" step="0.5" value="3">
    </div>
    <h2>标定状态</h2>
    <div class="stat" style="padding:8px 10px">
      <div class="k">椭圆拟合校正</div>
      <div class="v" id="s-calib" style="font-size:12px;color:var(--orange)">采集中…</div>
    </div>
  </aside>
  <main>
    <div class="stats">
      <div class="stat"><div class="k">真实位移</div><div class="v"><span id="s-xt">0.000</span><span class="u">μm</span></div></div>
      <div class="stat"><div class="k">测量位移</div><div class="v" style="color:var(--green)"><span id="s-xm">0.000</span><span class="u">μm</span></div></div>
      <div class="stat"><div class="k">测量误差</div><div class="v" style="color:var(--orange)"><span id="s-err">0.0</span><span class="u">nm</span></div></div>
      <div class="stat"><div class="k">解包裹相位</div><div class="v" style="color:var(--purple)"><span id="s-phi">0</span><span class="u">rad</span></div></div>
      <div class="stat"><div class="k">累计周期</div><div class="v"><span id="s-cyc">0</span><span class="u">T</span></div></div>
      <div class="stat"><div class="k">理论分辨率</div><div class="v" style="color:var(--cyan)"><span id="s-res">0.00</span><span class="u">nm</span></div></div>
    </div>
    <div class="charts">
      <div class="card"><div class="card-h"><span>李萨如图 (I – Q 椭圆)</span><span class="legend"><span><i style="background:#58a6ff"></i>原始信号</span></span></div><canvas id="cv-liss"></canvas></div>
      <div class="card"><div class="card-h"><span>正交信号波形</span><span class="legend"><span><i style="background:#58a6ff"></i>I</span><span><i style="background:#3fb950"></i>Q</span></span></div><canvas id="cv-wave"></canvas></div>
      <div class="card"><div class="card-h"><span>相位解包裹</span><span class="legend"><span><i style="background:#d29922"></i>原始 φ (−π,π]</span><span><i style="background:#bc8cff"></i>解包裹 φ</span></span></div><canvas id="cv-phase"></canvas></div>
      <div class="card"><div class="card-h"><span>位移测量</span><span class="legend"><span><i style="background:#8b949e"></i>真实值</span><span><i style="background:#3fb950"></i>测量值</span></span></div><canvas id="cv-disp"></canvas></div>
    </div>
  </main>
</div>
<script>
"use strict";
const MAXPTS = 1400;
const S = { running: false, timer: null, t: [], I: [], Q: [], pr: [], pu: [], xt: [], xm: [], calibrated: false };
function rc(id){ return document.getElementById(id); }
function readControls(){
  return {
    mode: rc('p-mode').value, d: +rc('p-d').value, N: +rc('p-N').value,
    theta: +rc('p-theta').value, velocity: +rc('p-v').value,
    noise: +rc('p-noise').value, dcI: +rc('p-dci').value,
    dcQ: +rc('p-dcq').value, gain: +rc('p-gain').value,
    ortho: +rc('p-ortho').value,
  };
}
const LABELS = {
  'p-d': v => v.toFixed(2) + ' μm', 'p-N': v => v + ' ×',
  'p-theta': v => v + ' mrad', 'p-v': v => v + ' μm/s',
  'p-noise': v => v.toFixed(3), 'p-dci': v => v.toFixed(2),
  'p-dcq': v => v.toFixed(2), 'p-gain': v => v.toFixed(2),
  'p-ortho': v => v.toFixed(1) + '°',
};
Object.keys(LABELS).forEach(id => {
  const el = rc(id); const tag = rc('v-' + id.slice(2));
  const upd = () => { if (tag) tag.textContent = LABELS[id](+el.value); };
  el.addEventListener('input', upd); upd();
});
['p-mode','p-d','p-N','p-theta','p-v','p-noise','p-dci','p-dcq','p-gain','p-ortho'].forEach(id => {
  rc(id).addEventListener('change', async () => { await fetch('/api/reset', {method:'POST'}); clearData(); });
});
function clearData(){
  S.t = []; S.I = []; S.Q = []; S.pr = []; S.pu = []; S.xt = []; S.xm = []; S.calibrated = false;
  rc('s-calib').textContent = '采集中…'; rc('s-calib').style.color = 'var(--orange)';
  render();
}
function appendData(d){
  const push = (arr, src) => { for (let i = 0; i < src.length; i++) arr.push(src[i]); if (arr.length > MAXPTS) arr.splice(0, arr.length - MAXPTS); };
  push(S.t, d.t); push(S.I, d.I); push(S.Q, d.Q); push(S.pr, d.pr); push(S.pu, d.pu); push(S.xt, d.xt); push(S.xm, d.xm);
  S.calibrated = d.calibrated;
  if (d.calibrated){ rc('s-calib').textContent = '✓ 已完成'; rc('s-calib').style.color = 'var(--green)'; }
  else { rc('s-calib').textContent = '采集中 ' + Math.round(d.calibProgress * 100) + '%'; }
  const resNm = d.scale * 1e-3 * 1e9;
  rc('s-res').textContent = resNm.toFixed(3);
}
function fitCanvas(canvas){
  const dpr = window.devicePixelRatio || 1; const w = canvas.clientWidth || 300; const h = canvas.clientHeight || 180;
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (canvas.width !== W || canvas.height !== H){ canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return {ctx, w, h};
}
function fmt(v){
  const a = Math.abs(v); if (a === 0) return '0';
  if (a >= 1e4 || a < 1e-3) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0); if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2); return v.toFixed(3);
}
function lineChart(canvas, series, opt){
  opt = opt || {}; const {ctx, w, h} = fitCanvas(canvas); ctx.clearRect(0, 0, w, h);
  const padL = opt.padL || 52, padR = 10, padT = 12, padB = 18;
  const pw = w - padL - padR, ph = h - padT - padB;
  if (pw <= 4 || ph <= 4) return;
  let n = 0, ymin = Infinity, ymax = -Infinity;
  for (const s of series){
    if (!s.y || !s.y.length) continue; n = Math.max(n, s.y.length);
    for (let i = 0; i < s.y.length; i++){ const v = s.y[i]; if (Number.isFinite(v)){ if (v < ymin) ymin = v; if (v > ymax) ymax = v; } }
  }
  if (!Number.isFinite(ymin)){ ymin = 0; ymax = 1; }
  if (ymax - ymin < 1e-15){ ymax = ymin + 1; }
  const mg = (ymax - ymin) * 0.12; ymin -= mg; ymax += mg;
  const X = i => padL + (n <= 1 ? pw * 0.5 : (i / (n - 1)) * pw);
  const Y = v => padT + (1 - (v - ymin) / (ymax - ymin)) * ph;
  ctx.strokeStyle = 'rgba(48,54,61,0.65)'; ctx.lineWidth = 1; ctx.beginPath();
  for (let k = 0; k <= 4; k++){ const y = padT + (k / 4) * ph; ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); }
  for (let k = 0; k <= 5; k++){ const x = padL + (k / 5) * pw; ctx.moveTo(x, padT); ctx.lineTo(x, padT + ph); }
  ctx.stroke();
  ctx.fillStyle = '#8b949e'; ctx.font = '10px ui-monospace,monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let k = 0; k <= 4; k++){ const v = ymax - (k / 4) * (ymax - ymin); ctx.fillText(fmt(v), padL - 5, padT + (k / 4) * ph); }
  if (ymin < 0 && ymax > 0){ ctx.strokeStyle = 'rgba(139,148,158,0.35)'; ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(padL + pw, Y(0)); ctx.stroke(); }
  for (const s of series){
    if (!s.y || s.y.length < 1) continue; const N = s.y.length; ctx.beginPath();
    for (let i = 0; i < N; i++){ const px = (N <= 1) ? padL + pw * 0.5 : padL + (i / (N - 1)) * pw; const py = Y(s.y[i]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.4; ctx.lineJoin = 'round'; ctx.stroke();
  }
}
function drawLissajous(){
  const canvas = rc('cv-liss'); const {ctx, w, h} = fitCanvas(canvas); ctx.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2; const R = Math.min(w, h) / 2 - 22; if (R <= 0) return;
  const n = S.I.length; const start = Math.max(0, n - 500);
  let amp = 0.1; for (let i = start; i < n; i++){ amp = Math.max(amp, Math.abs(S.I[i]), Math.abs(S.Q[i])); }
  const sc = (R * 0.85) / amp;
  ctx.strokeStyle = 'rgba(48,54,61,0.8)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R * 0.85, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = 'rgba(48,54,61,0.8)'; ctx.beginPath(); ctx.moveTo(cx - R - 6, cy); ctx.lineTo(cx + R + 6, cy); ctx.moveTo(cx, cy - R - 6); ctx.lineTo(cx, cy + R + 6); ctx.stroke();
  ctx.fillStyle = 'rgba(88,166,255,0.75)'; for (let i = start; i < n; i++){ const x = cx + S.I[i] * sc; const y = cy - S.Q[i] * sc; ctx.beginPath(); ctx.arc(x, y, 1.3, 0, Math.PI * 2); ctx.fill(); }
  ctx.fillStyle = '#8b949e'; ctx.font = '10px ui-monospace,monospace'; ctx.textAlign = 'center'; ctx.fillText('I', cx + R + 2, cy + 14); ctx.fillText('Q', cx - 12, cy - R - 2);
}
function render(){
  drawLissajous();
  lineChart(rc('cv-wave'), [{y: S.I, color: '#58a6ff'}, {y: S.Q, color: '#3fb950'}]);
  lineChart(rc('cv-phase'), [{y: S.pr, color: '#d29922', width: 1.0}, {y: S.pu, color: '#bc8cff', width: 1.6}]);
  const xt = S.xt.map(v => v * 1e9); const xm = S.xm.map(v => v * 1e9);
  lineChart(rc('cv-disp'), [{y: xt, color: '#8b949e', width: 1.2}, {y: xm, color: '#3fb950', width: 1.8}]);
  updateStats();
}
function updateStats(){
  const n = S.xt.length; if (!n) return;
  const xt = S.xt[n - 1], xm = S.xm[n - 1];
  rc('s-xt').textContent = (xt * 1e6).toFixed(4);
  rc('s-xm').textContent = (xm * 1e6).toFixed(4);
  rc('s-err').textContent = ((xm - xt) * 1e9).toFixed(2);
  rc('s-phi').textContent = S.pu[n - 1].toFixed(2);
  rc('s-cyc').textContent = (S.pu[n - 1] / (2 * Math.PI)).toFixed(2);
}
async function step(){
  if (!S.running) return;
  try {
    const r = await fetch('/api/step', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(readControls()) });
    const d = await r.json(); appendData(d); render();
  } catch (e){ console.error(e); }
  if (S.running) S.timer = setTimeout(step, 55);
}
rc('btn-run').addEventListener('click', async () => {
  if (S.running){ S.running = false; clearTimeout(S.timer); rc('btn-run').textContent = '▶ 开始'; rc('btn-run').classList.remove('on'); rc('status').textContent = '已停止'; rc('status').classList.remove('on'); }
  else { S.running = true; rc('btn-run').textContent = '⏸ 暂停'; rc('btn-run').classList.add('on'); rc('status').textContent = '运行中'; rc('status').classList.add('on'); step(); }
});
rc('btn-reset').addEventListener('click', async () => { await fetch('/api/reset', {method:'POST'}); clearData(); });
window.addEventListener('resize', () => render());
clearData();
</script>
</body>
</html>
"""

import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)