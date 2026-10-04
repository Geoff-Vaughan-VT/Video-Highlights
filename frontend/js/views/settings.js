// Hardware & settings: GPU / CUDA status, ffmpeg encoders and hwaccels,
// storage locations and disk space, versions, theme.

import { get } from '../api.js';
import { icon } from '../icons.js';
import { $, emptyState, esc, fmtBytes, isCurrent, routeToken, setMain, skLines } from '../ui.js';

const ENCODER_LABELS = {
  libx264: 'H.264 (CPU, libx264)', libx265: 'HEVC (CPU, libx265)', h264_nvenc: 'H.264 NVENC (NVIDIA)', hevc_nvenc: 'HEVC NVENC (NVIDIA)',
  av1_nvenc: 'AV1 NVENC (NVIDIA)', h264_videotoolbox: 'H.264 VideoToolbox (Apple)', hevc_videotoolbox: 'HEVC VideoToolbox (Apple)',
  h264_qsv: 'H.264 Quick Sync (Intel)', h264_vaapi: 'H.264 VA-API', h264_amf: 'H.264 AMF (AMD)',
};

const yes = (ok, label = ok ? 'Available' : 'Not available') => `<span class="badge ${ok ? 'ok' : ''}">${icon(ok ? 'ok' : 'x', 'sm')}${label}</span>`;

function gpuPanel(gpu) {
  if (!gpu) return `<div class="panel"><div class="panel-title"><span class="ttl-i">${icon('gpu', 'sm')} GPU</span></div><div class="note">GPU status unavailable.</div></div>`;
  const torch = gpu.torch || {};
  const smi = gpu.nvidia_smi || {};
  return `<div class="panel"><div class="charthead"><div class="panel-title" style="margin:0"><span class="ttl-i">${icon('gpu', 'sm')} GPU &amp; acceleration</span></div>
      ${gpu.ready ? '<span class="badge ok"><span class="dot"></span>GPU ready</span>' : '<span class="badge warn"><span class="dot"></span>CPU mode</span>'}</div>
    <p class="muted small" style="margin-bottom:12px">${esc(gpu.recommendation || '')}</p>
    <dl class="kv">
      <dt>PyTorch</dt><dd>${torch.installed ? esc(torch.version) : 'not installed'}${torch.cuda_version ? ` · CUDA ${esc(torch.cuda_version)}` : ''}</dd>
      <dt>CUDA devices</dt><dd>${torch.device_count ? esc((torch.devices || []).join(', ')) : 'none'}</dd>
      <dt>Apple MPS</dt><dd>${torch.mps_available ? 'available' : '—'}</dd>
      <dt>nvidia-smi</dt><dd>${smi.available ? 'ok' : esc(smi.error || 'unavailable')}</dd>
      <dt>NVENC rendering</dt><dd>${yes(!!gpu.rendering_ready, gpu.rendering_ready ? 'Ready' : 'Not ready')}</dd>
    </dl>
    ${(smi.gpus || []).map((g) => `<div class="divider"></div><div class="charthead"><b>${esc(g.name)}</b><span class="faint xs">driver ${esc(g.driver_version)}</span></div>
      <div class="kpis" style="grid-template-columns:repeat(3,1fr);margin:0">
        <div class="kpi"><div class="l">Memory</div><div class="v" style="font-size:18px">${Math.round((g.memory_used_mb || 0) / 1024)}<small>/ ${Math.round((g.memory_total_mb || 0) / 1024)} GB</small></div></div>
        <div class="kpi"><div class="l">Utilisation</div><div class="v" style="font-size:18px">${g.utilization_gpu_percent ?? '—'}<small>%</small></div></div>
        <div class="kpi"><div class="l">Temperature</div><div class="v" style="font-size:18px">${g.temperature_c ?? '—'}<small>°C</small></div></div></div>`).join('')}
  </div>`;
}

function ffmpegPanel(ff) {
  const enc = ff?.encoders || {};
  return `<div class="panel"><div class="panel-title"><span class="ttl-i">${icon('film', 'sm')} FFmpeg</span><span class="faint xs">encoders compiled into this build</span></div>
    <dl class="kv" style="margin-bottom:12px"><dt>Binary</dt><dd class="mono">${esc(ff?.path || 'not found')}</dd><dt>Version</dt><dd>${esc(ff?.version || '—')}</dd>
      <dt>HW decoders</dt><dd>${(ff?.hwaccels || []).length ? esc(ff.hwaccels.join(', ')) : 'none'}</dd></dl>
    <table><tbody>${Object.entries(ENCODER_LABELS).map(([key, label]) => `<tr><td>${esc(label)}</td><td class="r">${yes(!!enc[key], enc[key] ? 'In build' : 'Not in build')}</td></tr>`).join('')}</tbody></table>
    <div class="note">Hardware encoders also need the matching GPU and driver at run time — see GPU &amp; acceleration.</div></div>`;
}

function storagePanel(sys) {
  const disk = sys.disk;
  const used = disk ? disk.used_bytes / disk.total_bytes : 0;
  return `<div class="panel"><div class="panel-title"><span class="ttl-i">${icon('disk', 'sm')} Storage</span></div>
    <dl class="kv"><dt>Run outputs</dt><dd class="mono">${esc(sys.output_root)}</dd>
      <dt>Uploads</dt><dd class="mono">${esc(sys.local_storage_root)} (${esc(sys.storage_backend)})</dd></dl>
    ${disk ? `<div style="margin-top:16px"><div class="charthead"><span class="small">Disk</span><span class="small num">${fmtBytes(disk.free_bytes)} free of ${fmtBytes(disk.total_bytes)}</span></div>
      <div class="bar ${used > 0.9 ? 'danger' : used > 0.75 ? 'warn' : ''}"><i style="width:${Math.round(used * 100)}%"></i></div>
      ${disk.free_bytes < 50 * 1024 ** 3 ? '<div class="warnnote">Less than 50 GB free — a 4K match plus its proxy and renders can need 20–40 GB.</div>' : ''}</div>` : ''}</div>`;
}

export async function renderSettings() {
  const tok = routeToken();
  setMain(`<div class="page"><div class="pagehead"><div><div class="eyebrow">System</div><h1>Hardware &amp; settings</h1>
    <div class="sub">What this machine can accelerate, where files go, and versions.</div></div></div>
    <div class="cols-2" id="sysgrid"><div class="panel">${skLines(6)}</div><div class="panel">${skLines(6)}</div></div></div>`);
  const [sys, gpu] = await Promise.all([get('/studio/system').catch((e) => ({ error: e.message })), get('/health/gpu').catch(() => null)]);
  if (!isCurrent(tok)) return;
  if (sys.error) { $('#sysgrid').outerHTML = emptyState('alert', 'Could not read system info', esc(sys.error)); return; }
  $('#sysgrid').innerHTML = `<div>${gpuPanel(gpu)}${storagePanel(sys)}</div>
    <div>${ffmpegPanel(sys.ffmpeg)}
      <div class="panel"><div class="panel-title"><span class="ttl-i">${icon('info', 'sm')} About</span></div><dl class="kv">
        <dt>Product</dt><dd>Video Highlights Studio</dd><dt>API version</dt><dd>${esc(sys.api_version)}</dd><dt>Python</dt><dd>${esc(sys.python)}</dd>
        <dt>Job execution</dt><dd>${esc(sys.job_execution_mode)}</dd><dt>Auth required</dt><dd>${sys.auth_required ? 'yes' : 'no (development mode)'}</dd>
        <dt>API reference</dt><dd><a href="/docs" target="_blank" rel="noopener">/docs ${icon('external', 'sm')}</a></dd></dl></div></div>`;
}
