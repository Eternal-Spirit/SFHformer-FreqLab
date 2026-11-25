import { useState, useEffect, useRef } from 'react';
import {
  Upload, Wand2, RefreshCw, Zap, CheckCircle2, AlertTriangle,
  Sliders, Layers, Download, Eye, Sparkles, Gauge, ArrowRight,
  Search, ShieldAlert, Cpu, Dna, Radio, Image as ImageIcon
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:5001';

const NOISE_MODES = [
  { id: 'sinusoid', label: 'Single Sinusoid', icon: Radio, desc: 'Pure single wave-vector periodic stripe' },
  { id: 'multi_sinusoid', label: '2D Moiré / Grid', icon: Sparkles, desc: 'Multi-frequency harmonic cross-grid' },
  { id: 'mixed_sin_gauss', label: 'Sinusoid + Gaussian', icon: Layers, desc: 'Periodic wave + CMOS sensor noise' },
  { id: 'mixed_all', label: 'Full Multi-Noise', icon: ShieldAlert, desc: 'Sinusoid + Gaussian + Impulse noise' },
  { id: 'none', label: 'Uploaded As-Is', icon: ImageIcon, desc: 'No synthetic noise added (real photo)' },
];

const USE_CASE_SHORTCUTS = [
  { label: '📱 Screen Moiré', path: '/use_cases/moire_screen.png', mode: 'multi_sinusoid', k: 28, amp: 0.16, theta: 25 },
  { label: '🩻 MRI RF Stripe', path: '/use_cases/mri_rf_striping.png', mode: 'sinusoid', k: 48, amp: 0.22, theta: 45 },
  { label: '🛰️ Satellite Striping', path: '/use_cases/satellite_striping.png', mode: 'mixed_sin_gauss', k: 64, amp: 0.18, theta: 0 },
  { label: '📰 Halftone Descreen', path: '/use_cases/halftone_print.png', mode: 'multi_sinusoid', k: 36, amp: 0.15, theta: 45 },
  { label: '🔬 Microscopy 60Hz', path: '/use_cases/microscopy_hum.png', mode: 'mixed_sin_gauss', k: 22, amp: 0.20, theta: 90 },
];

const MODELS_LIST = [
  { id: 'adaptive_hybrid', name: 'Adaptive Hybrid', type: 'hybrid', badge: 'Auto Notch + SFHformer', color: '#ec4899' },
  { id: 'r1_baseline', name: 'R1 Baseline', type: 'neural', badge: 'Neural SFHformer-M', color: '#6366f1' },
  { id: 'r2_gated', name: 'R2 Gated', type: 'neural', badge: 'SFHformer + Radial Gate', color: '#8b5cf6' },
  { id: 'notch', name: 'Blind FFT Notch', type: 'classical', badge: 'Spectral Notch Filter', color: '#10b981' },
  { id: 'gaussian_s1.0', name: 'Gaussian Blur (σ=1)', type: 'classical', badge: 'Low-Pass Filter', color: '#f59e0b' },
  { id: 'median_3', name: 'Median 3×3', type: 'classical', badge: 'Non-Linear Filter', color: '#ef4444' },
];

export default function ImageRestoration() {
  const [serverHealthy, setServerHealthy] = useState(false);
  const [gpuName, setGpuName] = useState('Detecting...');
  const [currentImageSrc, setCurrentImageSrc] = useState('/samples/0807.png');
  const [currentBase64, setCurrentBase64] = useState(null);
  
  // Mixed Noise Controls
  const [noiseMode, setNoiseMode] = useState('mixed_all');
  const [frequencyK, setFrequencyK] = useState(25);
  const [amplitude, setAmplitude] = useState(0.15);
  const [orientationDeg, setOrientationDeg] = useState(0);
  const [gaussSigma, setGaussSigma] = useState(0.06);
  const [spDensity, setSpDensity] = useState(0.01);
  const [phase, setPhase] = useState(0);
  
  // Model & Processing
  const [selectedModel, setSelectedModel] = useState('adaptive_hybrid');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  
  // Results & Detection
  const [singleResult, setSingleResult] = useState(null);
  const [compareResult, setCompareResult] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [activeTab, setActiveTab] = useState('split');
  const [splitPos, setSplitPos] = useState(50);
  const [errorMsg, setErrorMsg] = useState(null);

  const fileInputRef = useRef(null);
  const splitContainerRef = useRef(null);

  // Check backend server health
  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        setServerHealthy(true);
        setGpuName(data.gpu || 'GPU');
        setErrorMsg(null);
      } else {
        setServerHealthy(false);
      }
    } catch {
      setServerHealthy(false);
      setGpuName('Offline (Start serve_restore.py)');
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  // Convert current image url to base64
  useEffect(() => {
    if (!currentImageSrc) return;
    const img = new window.Image();
    img.crossOrigin = 'Anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 256;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 256, 256);
      const b64 = canvas.toDataURL('image/png');
      setCurrentBase64(b64);
    };
    img.src = currentImageSrc;
  }, [currentImageSrc]);

  // Handle custom file upload
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      setCurrentImageSrc(event.target.result);
      setNoiseMode('none'); // Default to processing uploaded image directly
      setSingleResult(null);
      setCompareResult(null);
      setDetectionResult(null);
    };
    reader.readAsDataURL(file);
  };

  // Run Noise Detection independently
  const runDetection = async () => {
    if (!currentBase64) return;
    setIsDetecting(true);
    try {
      const res = await fetch(`${API_BASE}/api/detect_noise`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: singleResult ? singleResult.corrupted_img : currentBase64 }),
      });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setDetectionResult(data);
    } catch (err) {
      setErrorMsg(`Detection failed: ${err.message}`);
    } finally {
      setIsDetecting(false);
    }
  };

  // Run restoration
  const runRestoration = async () => {
    if (!currentBase64) return;
    setIsProcessing(true);
    setErrorMsg(null);

    const thetaRad = (orientationDeg * Math.PI) / 180.0;
    const payload = {
      image: currentBase64,
      add_noise: noiseMode !== 'none',
      noise_mode: noiseMode,
      k: frequencyK,
      amp: amplitude,
      theta: thetaRad,
      gauss_sigma: gaussSigma,
      sp_density: spDensity,
      phase: phase,
      method: selectedModel,
    };

    try {
      if (compareMode) {
        const res = await fetch(`${API_BASE}/api/restore_all`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();
        setCompareResult(data);
        if (data.noise_detection) setDetectionResult(data.noise_detection);
      } else {
        const res = await fetch(`${API_BASE}/api/restore`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();
        setSingleResult(data);
        if (data.noise_detection) setDetectionResult(data.noise_detection);
      }
    } catch (err) {
      setErrorMsg(`Inference request failed: ${err.message}. Ensure Python server is running on port 5001.`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Auto-run on first load once base64 is ready
  useEffect(() => {
    if (currentBase64 && !singleResult && serverHealthy) {
      runRestoration();
    }
  }, [currentBase64, serverHealthy]);

  const isInBand = frequencyK >= 8 && frequencyK <= 38;

  const handleSplitMouseMove = (e) => {
    if (!splitContainerRef.current) return;
    const rect = splitContainerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setSplitPos((x / rect.width) * 100);
  };

  const applyUseCasePreset = (uc) => {
    setCurrentImageSrc(uc.path);
    setNoiseMode(uc.mode);
    setFrequencyK(uc.k);
    setAmplitude(uc.amp);
    setOrientationDeg(uc.theta);
    setSingleResult(null);
    setCompareResult(null);
  };

  return (
    <div className="page">
      {/* Header */}
      <div className="page__header">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h1 className="page__title">Mixed Noise & Image Restoration Studio</h1>
            <span className={`badge ${serverHealthy ? 'badge--primary' : 'badge--secondary'}`}>
              <Zap size={14} /> {serverHealthy ? `GPU Active (${gpuName})` : 'Backend Disconnected'}
            </span>
          </div>
          <p className="page__description">
            Detect and clear mixed noise contaminations (sinusoidal moiré, additive Gaussian, and impulse noise) using the Adaptive Hybrid Pipeline and SFHformer neural models.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className={`btn ${compareMode ? 'btn--primary' : 'btn--secondary'}`}
            onClick={() => setCompareMode(!compareMode)}
          >
            <Layers size={16} /> {compareMode ? 'Single Model Mode' : 'Compare All Models'}
          </button>
          <button
            className="btn btn--primary"
            onClick={runRestoration}
            disabled={isProcessing || !currentBase64}
          >
            {isProcessing ? <RefreshCw size={16} className="spin" /> : <Wand2 size={16} />}
            {isProcessing ? 'Restoring on GPU...' : 'Detect & Restore'}
          </button>
        </div>
      </div>

      {/* Real-World Use Case Preset Shortcuts */}
      <div className="use-case-shortcuts-bar">
        <span className="use-case-shortcuts-label"><Cpu size={14} /> Industry Presets:</span>
        <div className="use-case-shortcuts-chips">
          {USE_CASE_SHORTCUTS.map((uc, i) => (
            <button
              key={i}
              className="chip chip--shortcut"
              onClick={() => applyUseCasePreset(uc)}
            >
              {uc.label}
            </button>
          ))}
        </div>
      </div>

      {errorMsg && (
        <div className="alert-box alert-box--warning">
          <AlertTriangle size={18} />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Studio Grid */}
      <div className="studio-layout">
        {/* Left Column: Controls */}
        <div className="studio-sidebar">
          {/* Section 1: Image Source */}
          <div className="studio-card">
            <h3 className="studio-card__title">
              <Upload size={16} /> 1. Image Source
            </h3>

            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              accept="image/*"
              onChange={handleFileUpload}
            />

            <button
              className="btn btn--secondary btn--full"
              onClick={() => fileInputRef.current?.click()}
              style={{ marginBottom: '12px' }}
            >
              <Upload size={16} /> Upload Custom Photo / Scan
            </button>

            <span className="studio-card__sublabel">Or Pick Benchmark Crop:</span>
            <div className="preset-crops-grid">
              {['0807', '0809', '0810', '0812'].map((id) => (
                <button
                  key={id}
                  className={`preset-crop-item ${currentImageSrc.includes(id) ? 'preset-crop-item--active' : ''}`}
                  onClick={() => {
                    setCurrentImageSrc(`/samples/${id}.png`);
                    setSingleResult(null);
                    setCompareResult(null);
                  }}
                >
                  <img src={`/samples/${id}.png`} alt={`DIV2K ${id}`} className="preset-crop-thumb" />
                  <span>Crop {id}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Section 2: Noise Mode & Mixed Interference */}
          <div className="studio-card">
            <h3 className="studio-card__title">
              <Sliders size={16} /> 2. Noise & Corruption Profile
            </h3>

            {/* Mode selection chips */}
            <div className="noise-mode-chips">
              {NOISE_MODES.map((m) => {
                const Icon = m.icon;
                return (
                  <button
                    key={m.id}
                    className={`noise-mode-btn ${noiseMode === m.id ? 'noise-mode-btn--active' : ''}`}
                    onClick={() => setNoiseMode(m.id)}
                    title={m.desc}
                  >
                    <Icon size={14} />
                    <span>{m.label}</span>
                  </button>
                );
              })}
            </div>

            {noiseMode !== 'none' ? (
              <div style={{ marginTop: '14px' }}>
                {/* Frequency slider */}
                <div className="control-slider">
                  <div className="control-slider__header">
                    <span>Sinusoid / Moiré Frequency (k): <strong>{frequencyK}</strong></span>
                    <span className={`band-badge ${isInBand ? 'band-badge--in' : 'band-badge--out'}`}>
                      {isInBand ? 'In-Band' : 'Out-of-Band'}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="3"
                    max="120"
                    step="1"
                    value={frequencyK}
                    onChange={(e) => setFrequencyK(Number(e.target.value))}
                    className="slider-input"
                  />
                </div>

                {/* Amplitude slider */}
                <div className="control-slider">
                  <div className="control-slider__header">
                    <span>Wave Amplitude (A): <strong>{amplitude.toFixed(2)}</strong></span>
                  </div>
                  <input
                    type="range"
                    min="0.02"
                    max="0.40"
                    step="0.01"
                    value={amplitude}
                    onChange={(e) => setAmplitude(Number(e.target.value))}
                    className="slider-input"
                  />
                </div>

                {/* Gaussian Noise slider (when enabled) */}
                {(noiseMode === 'mixed_sin_gauss' || noiseMode === 'mixed_all') && (
                  <div className="control-slider">
                    <div className="control-slider__header">
                      <span>Gaussian Sensor Noise (σ): <strong>{gaussSigma.toFixed(3)}</strong></span>
                    </div>
                    <input
                      type="range"
                      min="0.01"
                      max="0.15"
                      step="0.005"
                      value={gaussSigma}
                      onChange={(e) => setGaussSigma(Number(e.target.value))}
                      className="slider-input"
                    />
                  </div>
                )}

                {/* Impulse Noise slider (when enabled) */}
                {noiseMode === 'mixed_all' && (
                  <div className="control-slider">
                    <div className="control-slider__header">
                      <span>Impulse / Salt & Pepper: <strong>{(spDensity * 100).toFixed(1)}%</strong></span>
                    </div>
                    <input
                      type="range"
                      min="0.002"
                      max="0.05"
                      step="0.002"
                      value={spDensity}
                      onChange={(e) => setSpDensity(Number(e.target.value))}
                      className="slider-input"
                    />
                  </div>
                )}

                {/* Orientation slider */}
                <div className="control-slider">
                  <div className="control-slider__header">
                    <span>Wave Orientation (θ): <strong>{orientationDeg}°</strong></span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="180"
                    step="5"
                    value={orientationDeg}
                    onChange={(e) => setOrientationDeg(Number(e.target.value))}
                    className="slider-input"
                  />
                </div>
              </div>
            ) : (
              <p className="studio-card__hint" style={{ marginTop: '10px' }}>
                Synthetic noise disabled. The AI Noise Detector will analyze the uploaded image directly for real-world artifacts.
              </p>
            )}
          </div>

          {/* Section 3: Model Selection */}
          {!compareMode && (
            <div className="studio-card">
              <h3 className="studio-card__title">
                <Layers size={16} /> 3. Restoration Engine
              </h3>
              <div className="model-select-list">
                {MODELS_LIST.map((m) => (
                  <button
                    key={m.id}
                    className={`model-select-item ${selectedModel === m.id ? 'model-select-item--active' : ''}`}
                    onClick={() => setSelectedModel(m.id)}
                  >
                    <div className="model-select-item__header">
                      <span className="method-dot" style={{ background: m.color }}></span>
                      <strong>{m.name}</strong>
                    </div>
                    <span className="model-select-item__badge">{m.badge}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Visualizer & Noise Diagnosis */}
        <div className="studio-main">
          {/* Automated AI Noise Detection Banner */}
          {detectionResult && (
            <div className="detection-banner">
              <div className="detection-banner__header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Search size={18} className="detection-banner__icon" />
                  <h4>Automated Spectral & Spatial Noise Diagnosis</h4>
                </div>
                <button
                  className="btn btn--secondary btn--sm"
                  onClick={runDetection}
                  disabled={isDetecting}
                >
                  <RefreshCw size={12} className={isDetecting ? 'spin' : ''} /> Re-scan
                </button>
              </div>

              {/* Detected Noise Types Chips */}
              <div className="detection-types-list">
                {detectionResult.detected_types.map((typeStr, idx) => (
                  <span key={idx} className="detection-type-chip">
                    <CheckCircle2 size={13} /> {typeStr}
                  </span>
                ))}
              </div>

              {/* Peak Details & Recommendations */}
              <div className="detection-details-grid">
                <div>
                  <span className="detail-label">Detected Spectral Peaks:</span>
                  {detectionResult.peaks.length > 0 ? (
                    <div className="detected-peaks-row">
                      {detectionResult.peaks.map((pk, pi) => (
                        <span key={pi} className="peak-badge">
                          k={pk.k_radial} @ {pk.theta_deg}° ({pk.strength_ratio}× floor)
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="detail-value">No discrete periodic spikes detected</span>
                  )}
                </div>

                <div>
                  <span className="detail-label">Estimated Noise Floor:</span>
                  <span className="detail-value mono">
                    Gaussian σ ≈ {detectionResult.sigma_estimated} · Impulse: {detectionResult.impulse_pct}%
                  </span>
                </div>
              </div>

              <div className="detection-recommendation">
                <Dna size={14} />
                <span>AI Recommendation: <strong>{detectionResult.recommendation}</strong></span>
              </div>
            </div>
          )}

          {/* View Mode Tabs */}
          <div className="studio-tabs">
            <button
              className={`chip ${activeTab === 'split' ? 'chip--active' : ''}`}
              onClick={() => setActiveTab('split')}
            >
              <Eye size={14} /> Split Comparison Slider
            </button>
            <button
              className={`chip ${activeTab === 'side' ? 'chip--active' : ''}`}
              onClick={() => setActiveTab('side')}
            >
              Side-by-Side View
            </button>
            <button
              className={`chip ${activeTab === 'spectra' ? 'chip--active' : ''}`}
              onClick={() => setActiveTab('spectra')}
            >
              <Sparkles size={14} /> 2D FFT Spectra Analysis
            </button>
          </div>

          {/* Single Model View */}
          {!compareMode && singleResult && (
            <>
              {/* Split Slider View */}
              {activeTab === 'split' && (
                <div className="split-view-container">
                  <div
                    className="split-viewer"
                    ref={splitContainerRef}
                    onMouseMove={handleSplitMouseMove}
                    onTouchMove={(e) => {
                      if (!splitContainerRef.current) return;
                      const rect = splitContainerRef.current.getBoundingClientRect();
                      const touch = e.touches[0];
                      const x = Math.max(0, Math.min(touch.clientX - rect.left, rect.width));
                      setSplitPos((x / rect.width) * 100);
                    }}
                  >
                    <img
                      src={singleResult.restored_img}
                      alt="Restored"
                      className="split-viewer__img split-viewer__img--restored"
                    />

                    <div
                      className="split-viewer__clip"
                      style={{ width: `${splitPos}%` }}
                    >
                      <img
                        src={singleResult.corrupted_img}
                        alt="Corrupted"
                        className="split-viewer__img split-viewer__img--corrupted"
                      />
                      <span className="split-viewer__tag split-viewer__tag--left">
                        Corrupted ({singleResult.metrics.psnr_corrupted} dB)
                      </span>
                    </div>

                    <span className="split-viewer__tag split-viewer__tag--right">
                      Restored with {singleResult.method} ({singleResult.metrics.psnr_restored} dB)
                    </span>

                    <div
                      className="split-viewer__handle"
                      style={{ left: `${splitPos}%` }}
                    >
                      <div className="split-viewer__handle-line"></div>
                      <div className="split-viewer__handle-knob">⇄</div>
                    </div>
                  </div>
                  <p className="split-view-hint">
                    Drag the slider to compare Corrupted vs. Restored output in real time.
                  </p>
                </div>
              )}

              {/* Side-by-Side View */}
              {activeTab === 'side' && (
                <div className="side-by-side-grid">
                  <div className="side-card">
                    <h4>Ground Truth / Clean</h4>
                    <img src={singleResult.clean_img} alt="Clean Reference" className="side-card__img" />
                    <span className="side-card__metric">Ground Truth Reference</span>
                  </div>
                  <div className="side-card">
                    <h4>Corrupted ({singleResult.noise_mode})</h4>
                    <img src={singleResult.corrupted_img} alt="Corrupted" className="side-card__img" />
                    <span className="side-card__metric">
                      PSNR: {singleResult.metrics.psnr_corrupted} dB · SSIM: {singleResult.metrics.ssim_corrupted}
                    </span>
                  </div>
                  <div className="side-card side-card--highlight">
                    <h4>Restored ({singleResult.method})</h4>
                    <img src={singleResult.restored_img} alt="Restored" className="side-card__img" />
                    <span className="side-card__metric positive">
                      PSNR: {singleResult.metrics.psnr_restored} dB · SSIM: {singleResult.metrics.ssim_restored}
                    </span>
                  </div>
                </div>
              )}

              {/* 2D FFT Spectra Analysis */}
              {activeTab === 'spectra' && singleResult.spectra && (
                <div className="side-by-side-grid">
                  <div className="side-card">
                    <h4>Clean 2D FFT Spectrum</h4>
                    <img src={singleResult.spectra.clean} alt="Clean Spectrum" className="side-card__img" />
                    <p className="side-card__desc">Natural smooth power distribution across frequency plane.</p>
                  </div>
                  <div className="side-card">
                    <h4>Corrupted 2D FFT Spectrum</h4>
                    <img src={singleResult.spectra.corrupted} alt="Corrupted Spectrum" className="side-card__img" />
                    <p className="side-card__desc">Notice distinct bright harmonic interference peaks.</p>
                  </div>
                  <div className="side-card side-card--highlight">
                    <h4>Restored 2D FFT Spectrum</h4>
                    <img src={singleResult.spectra.restored} alt="Restored Spectrum" className="side-card__img" />
                    <p className="side-card__desc">Interference impulses suppressed and high-frequency power normalized.</p>
                  </div>
                </div>
              )}

              {/* Quantitative Metrics Bar */}
              <div className="metrics-banner">
                <div className="metric-box">
                  <span className="metric-box__label">ΔPSNR Improvement</span>
                  <span className={`metric-box__value ${singleResult.metrics.delta_psnr >= 0 ? 'positive' : 'negative'}`}>
                    {singleResult.metrics.delta_psnr > 0 ? '+' : ''}{singleResult.metrics.delta_psnr} dB
                  </span>
                  <span className="metric-box__sub">
                    {singleResult.metrics.psnr_corrupted} dB → {singleResult.metrics.psnr_restored} dB
                  </span>
                </div>

                <div className="metric-box">
                  <span className="metric-box__label">ΔSSIM Structural Gain</span>
                  <span className="metric-box__value positive">
                    +{singleResult.metrics.delta_ssim.toFixed(4)}
                  </span>
                  <span className="metric-box__sub">
                    {singleResult.metrics.ssim_corrupted} → {singleResult.metrics.ssim_restored}
                  </span>
                </div>

                <div className="metric-box">
                  <span className="metric-box__label">Noise Regime</span>
                  <span className={`metric-box__value ${singleResult.in_band ? 'positive' : 'warning'}`}>
                    {singleResult.in_band ? 'In-Band' : 'Out-of-Band'}
                  </span>
                  <span className="metric-box__sub">Mode: {singleResult.noise_mode}</span>
                </div>

                <div className="metric-box">
                  <span className="metric-box__label">GPU Execution Speed</span>
                  <span className="metric-box__value mono">
                    {singleResult.metrics.latency_ms} ms
                  </span>
                  <span className="metric-box__sub">Total Roundtrip: {singleResult.metrics.total_ms} ms</span>
                </div>

                <div className="metric-box-action">
                  <a
                    href={singleResult.restored_img}
                    download={`restored_${singleResult.method}.png`}
                    className="btn btn--secondary"
                  >
                    <Download size={14} /> Download PNG
                  </a>
                </div>
              </div>
            </>
          )}

          {/* Compare Mode: All Models Side-by-Side */}
          {compareMode && compareResult && (
            <div className="compare-mode-container">
              <div className="compare-mode-header">
                <h3>Multi-Model Restoration on Identical Corrupted Input</h3>
                <span className="badge badge--primary">
                  Corrupted Input: {compareResult.baseline_metrics.psnr_corrupted} dB
                </span>
              </div>

              <div className="compare-models-grid">
                {Object.entries(compareResult.methods).map(([mid, mRes]) => {
                  const mInfo = MODELS_LIST.find((m) => m.id === mid) || { name: mid, color: '#6366f1' };
                  const isHybrid = mid === 'adaptive_hybrid';
                  return (
                    <div
                      key={mid}
                      className={`compare-model-card ${isHybrid ? 'compare-model-card--highlight' : ''}`}
                      style={{ '--card-accent': mInfo.color }}
                    >
                      <div className="compare-model-card__header">
                        <div>
                          <span className="method-dot" style={{ background: mInfo.color }}></span>
                          <strong>{mInfo.name}</strong>
                        </div>
                        <span className={`badge ${mRes.delta_psnr >= 8 ? 'badge--primary' : 'badge--secondary'}`}>
                          {mRes.delta_psnr > 0 ? '+' : ''}{mRes.delta_psnr} dB
                        </span>
                      </div>
                      <img src={mRes.restored_img} alt={mInfo.name} className="compare-model-card__img" />
                      <div className="compare-model-card__stats">
                        <div>
                          <span>PSNR</span>
                          <strong>{mRes.psnr} dB</strong>
                        </div>
                        <div>
                          <span>SSIM</span>
                          <strong>{mRes.ssim}</strong>
                        </div>
                        <div>
                          <span>Latency</span>
                          <strong className="mono">{mRes.latency_ms} ms</strong>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Empty State */}
          {!singleResult && !compareResult && (
            <div className="studio-empty">
              <Sparkles size={40} className="studio-empty__icon" />
              <h3>Ready to Detect & Restore</h3>
              <p>
                Select an image, configure mixed noise parameters, and click <strong>Detect & Restore</strong> to run on your GPU.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
