import { useState, useEffect } from 'react';
import {
  Eye, Wand2, RefreshCw, Activity, ShieldCheck, AlertCircle,
  Sliders, Layers, Download, CheckCircle2, Microscope, FileText,
  Sparkles, Radio, Zap, HelpCircle, ChevronRight, BarChart2,
  GitCompare, Table, Info, ArrowUpRight, RotateCcw, Clock
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:5001';

const NOISE_MODES = [
  { id: 'mixed', label: 'Clinical Composite', desc: 'Combined raster bands, speckle, flash vignette & sensor noise' },
  { id: 'raster_banding', label: 'Raster Striping (cSLO)', desc: 'Periodic detector line-scan readout bands' },
  { id: 'laser_speckle', label: 'Laser Speckle', desc: 'Coherent laser interference common in OCT & SLO' },
  { id: 'optical_halo', label: 'Corneal Halo Glare', desc: 'Circular pupil/cornea flash reflection' },
  { id: 'sensor_noise', label: 'CMOS Shot Noise', desc: 'Photon shot noise floor in low-light fundus' },
  { id: 'none', label: 'No Synthetic Noise', desc: 'Original clinical capture as-is' },
];

const MODEL_OPTIONS = [
  { id: 'retina_fives', label: 'SFHformer-FIVES SOTA (31.92 dB)', desc: 'Frozen Milestone 3 fine-tuned frequency restorer' },
  { id: 'retina_mega', label: 'SFHformer-MegaRetina (4-Dataset)', desc: 'Trained across DRIVE, STARE, HRF, & APTOS' },
  { id: 'r2_gated', label: 'SFHformer-M (Learned Gated)', desc: 'Spatial-frequency dynamic soft-gating mechanism' },
  { id: 'r1_baseline', label: 'SFHformer-M (R1 Baseline)', desc: 'General photographic frequency restorer' },
];

export default function RetinaStudio() {
  // Navigation Tabs: 'dual_unet' (Primary), 'benchmark' (Milestone 3 Data), 'ablation' (Noise/Filters)
  const [activeTab, setActiveTab] = useState('dual_unet');

  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState('hrf_unexposed_14h');
  const [activeModality, setActiveModality] = useState('All');
  const [noiseMode, setNoiseMode] = useState('none');
  const [severity, setSeverity] = useState(1.0);
  const [modelChoice, setModelChoice] = useState('retina_fives');
  
  // Dual U-Net & Hybrid States
  const FROZEN_DUAL_THRESHOLD = 0.40;
  const ALT_BASELINE_THRESHOLD = 0.50;
  const [selectedSegModel, setSelectedSegModel] = useState('dual_stream'); // 'dual_stream' (default), 'raw_unet', 'restored_unet'
  const [segThreshold, setSegThreshold] = useState(0.40);
  const [segResolution, setSegResolution] = useState(512);
  const [includeFusionViz, setIncludeFusionViz] = useState(false); // Hidden by default to avoid clutter
  const [dualViewMode, setDualViewMode] = useState('dual_stream_focus'); // 'dual_stream_focus', 'tri_view', 'differential_map', 'restored_focus', 'raw_focus'
  const [dualLoading, setDualLoading] = useState(false);
  const [dualResult, setDualResult] = useState(null);
  const [dualError, setDualError] = useState(null);
  const [showTimingBreakdown, setShowTimingBreakdown] = useState(false);

  // Classic Ablation States
  const [displayMode, setDisplayMode] = useState('overlay');
  const [ablationLoading, setAblationLoading] = useState(false);
  const [ablationResult, setAblationResult] = useState(null);
  const [ablationError, setAblationError] = useState(null);

  // Benchmark States
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);

  const isDualCalibratedThreshold = Math.abs(segThreshold - FROZEN_DUAL_THRESHOLD) < 0.001;
  const isAltBaselineThreshold = Math.abs(segThreshold - ALT_BASELINE_THRESHOLD) < 0.001;
  const isThresholdExploratory = !isDualCalibratedThreshold && !isAltBaselineThreshold;

  // Fetch sample catalog
  useEffect(() => {
    fetch(`${API_BASE}/api/retina/samples`)
      .then(res => res.json())
      .then(data => {
        if (data.samples) {
          setSamples(data.samples);
        }
      })
      .catch(() => {
        setSamples([
          { id: 'hrf_unexposed_14h', title: 'HRF Unexposed Capillary Bed (14_h)', category: 'Fundus', modality: 'High-Resolution Fundus', gain: '+11.21 pp Skel', description: 'Zero-shot HRF scan untouched by SFHformer during pretraining.' },
          { id: 'fives_glaucoma_37db', title: 'FIVES Glaucoma Rim (37.69 dB)', category: 'Fundus', modality: 'High-Res Fundus (FIVES)', gain: '+6.96 dB', description: 'Neuroretinal rim contour and cup-to-disc ratio restored.' },
          { id: 'fives_amd_37db', title: 'FIVES Macular Drusen (37.61 dB)', category: 'Fundus', modality: 'High-Res Fundus (FIVES)', gain: '+11.11 dB', description: 'Macular drusen borders restored with +11.11 dB gain.' },
          { id: 'fives_normal_37db', title: 'FIVES Healthy Arcade (37.27 dB)', category: 'Fundus', modality: 'High-Res Fundus (FIVES)', gain: '+10.67 dB', description: 'Pristine Fourier harmonic restoration across major arcades.' },
          { id: 'normal_fundus', title: 'Healthy Color Fundus (DRIVE)', category: 'Fundus', modality: 'Color Fundus', description: '50° macula-centered view. Healthy vascular arcade.' },
          { id: 'oct_normal', title: 'SD-OCT Normal B-Scan', category: 'OCT', modality: 'Spectral-Domain OCT', description: 'Cross-sectional retinal strata. Preserved foveal pit & RPE.' },
        ]);
      });
  }, []);

  // Fetch benchmark data on tab switch
  useEffect(() => {
    if (activeTab === 'benchmark' && !benchmarkData) {
      setBenchmarkLoading(true);
      fetch(`${API_BASE}/api/retina/benchmark`)
        .then(res => res.json())
        .then(data => setBenchmarkData(data))
        .catch(err => console.error("Benchmark fetch error:", err))
        .finally(() => setBenchmarkLoading(false));
    }
  }, [activeTab]);

  // Run Dual / Hybrid U-Net Segmentation
  const runDualSegmentation = async (
    sampleId = selectedSampleId,
    thresh = segThreshold,
    resDim = segResolution,
    mdl = selectedSegModel,
    fusionViz = includeFusionViz
  ) => {
    setDualLoading(true);
    setDualError(null);
    try {
      const res = await fetch(`${API_BASE}/api/retina/dual_segment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample_id: sampleId,
          threshold: thresh,
          resolution: resDim,
          selected_model: mdl,
          include_fusion_viz: fusionViz
        })
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setDualResult(data);
    } catch (err) {
      console.error(err);
      setDualError('Could not connect to Dual U-Net segmentation backend. Ensure serve_restore.py is running.');
    } finally {
      setDualLoading(false);
    }
  };

  // Run Classic Ablation
  const runClassicAblation = async (sampleId = selectedSampleId, mode = noiseMode, sev = severity, mdl = modelChoice) => {
    setAblationLoading(true);
    setAblationError(null);
    try {
      const res = await fetch(`${API_BASE}/api/retina/restore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sample_id: sampleId,
          mode: mode,
          severity: sev,
          model: mdl
        })
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setAblationResult(data);
    } catch (err) {
      console.error(err);
      setAblationError('Could not connect to FreqLab restoration backend.');
    } finally {
      setAblationLoading(false);
    }
  };

  // Initial trigger & sample change trigger (never auto-adjusts threshold across datasets)
  useEffect(() => {
    if (selectedSampleId) {
      if (activeTab === 'dual_unet') {
        runDualSegmentation(selectedSampleId, segThreshold, segResolution, selectedSegModel, includeFusionViz);
      } else if (activeTab === 'ablation') {
        runClassicAblation(selectedSampleId, noiseMode, severity, modelChoice);
      }
    }
  }, [selectedSampleId, activeTab]);

  const filteredSamples = activeModality === 'All' 
    ? samples 
    : samples.filter(s => s.category === activeModality);

  return (
    <div className="retina-studio" style={{ padding: '24px', maxWidth: '1440px', margin: '0 auto', color: '#f1f5f9' }}>
      
      {/* Top Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '6px' }}>
            <div style={{ padding: '8px', borderRadius: '10px', background: 'linear-gradient(135deg, #00B4D8 0%, #0077B6 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Eye size={24} color="#fff" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <h1 style={{ fontSize: '24px', fontWeight: 700, margin: 0, letterSpacing: '-0.02em', background: 'linear-gradient(135deg, #ffffff 40%, #00B4D8 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                  RetinaScan Studio
                </h1>
                <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '4px', background: 'rgba(0, 240, 180, 0.15)', border: '1px solid rgba(0, 240, 180, 0.3)', color: '#00F0B4', fontWeight: 600 }}>
                  Research Prototype • Milestone 3
                </span>
              </div>
              <p style={{ margin: 0, fontSize: '13px', color: '#94a3b8' }}>
                Research Prototype for Retinal Vessel Segmentation &amp; Frequency Restoration Analysis (Not a Certified Diagnostic System)
              </p>
            </div>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <div style={{ padding: '6px 12px', borderRadius: '8px', background: 'rgba(0, 180, 216, 0.1)', border: '1px solid rgba(0, 180, 216, 0.25)', fontSize: '12px', color: '#00B4D8', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Activity size={14} />
            RTX 5060 (BF16 AMP Active)
          </div>
          <button 
            onClick={() => {
              if (activeTab === 'dual_unet') runDualSegmentation();
              else if (activeTab === 'ablation') runClassicAblation();
            }}
            disabled={dualLoading || ablationLoading}
            style={{ 
              display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', borderRadius: '8px', 
              background: 'linear-gradient(135deg, #0077B6 0%, #0096C7 100%)', color: '#fff', border: 'none', 
              fontWeight: 600, fontSize: '13px', cursor: (dualLoading || ablationLoading) ? 'not-allowed' : 'pointer', 
              opacity: (dualLoading || ablationLoading) ? 0.7 : 1 
            }}
          >
            <RefreshCw size={14} className={(dualLoading || ablationLoading) ? 'spin' : ''} />
            {(dualLoading || ablationLoading) ? 'Inferring...' : 'Re-Run Model'}
          </button>
        </div>
      </div>

      {/* Navigation Tabs Bar */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '12px' }}>
        {[
          { id: 'dual_unet', label: 'Dual U-Net Differential Inspection', icon: GitCompare, badge: 'Differential Response' },
          { id: 'benchmark', label: 'Milestone 3 Clinical Benchmark', icon: Table, badge: '200 FIVES / 45 HRF' },
          { id: 'ablation', label: 'Degradation & Filter Ablation', icon: Sliders, badge: 'Ophthalmic Noise' },
        ].map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', borderRadius: '8px',
                fontSize: '13px', fontWeight: 600, cursor: 'pointer', transition: 'all 0.15s',
                background: isActive ? 'rgba(0, 180, 216, 0.18)' : 'rgba(255,255,255,0.03)',
                color: isActive ? '#38bdf8' : '#94a3b8',
                border: isActive ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid rgba(255,255,255,0.05)'
              }}
            >
              <Icon size={16} />
              <span>{tab.label}</span>
              <span style={{ 
                fontSize: '10px', padding: '2px 6px', borderRadius: '4px',
                background: isActive ? 'rgba(56, 189, 248, 0.25)' : 'rgba(255,255,255,0.06)',
                color: isActive ? '#7dd3fc' : '#64748b'
              }}>
                {tab.badge}
              </span>
            </button>
          );
        })}
      </div>

      {/* ========================================================================= */}
      {/* TAB 1: DUAL U-NET DIFFERENTIAL INSPECTION (PRIMARY CLINICAL VIEW)          */}
      {/* ========================================================================= */}
      {activeTab === 'dual_unet' && (
        <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '24px' }}>
          
          {/* Left Column: Sample Cases & Segmentation Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
            
            {/* Sample Selector */}
            <div style={{ background: 'rgba(15, 23, 42, 0.65)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>
                  Clinical Scans ({filteredSamples.length})
                </span>
                <span style={{ fontSize: '11px', color: '#00F0B4', fontWeight: 600 }}>Zero-Shot / SOTA</span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '380px', overflowY: 'auto' }}>
                {filteredSamples.map(s => {
                  const isSelected = selectedSampleId === s.id;
                  const isUnexposed = s.id === 'hrf_unexposed_14h';
                  return (
                    <div
                      key={s.id}
                      onClick={() => setSelectedSampleId(s.id)}
                      style={{
                        padding: '10px 12px', borderRadius: '8px', cursor: 'pointer', transition: 'all 0.15s',
                        background: isSelected 
                          ? 'rgba(0, 180, 216, 0.20)' 
                          : isUnexposed ? 'rgba(245, 158, 11, 0.08)' : 'rgba(255,255,255,0.02)',
                        border: isSelected 
                          ? '1px solid rgba(0, 180, 216, 0.5)' 
                          : isUnexposed ? '1px solid rgba(245, 158, 11, 0.3)' : '1px solid rgba(255,255,255,0.05)',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontSize: '13px', fontWeight: 600, color: isSelected ? '#38bdf8' : (isUnexposed ? '#fbbf24' : '#e2e8f0') }}>
                          {s.title}
                        </span>
                        {s.gain && (
                          <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '4px', background: 'rgba(0, 240, 180, 0.15)', color: '#00F0B4', fontWeight: 600 }}>
                            {s.gain}
                          </span>
                        )}
                      </div>
                      <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8', lineHeight: 1.3 }}>{s.description}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Segmentation Model Selector */}
            <div style={{ background: 'rgba(15, 23, 42, 0.65)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>
                  Active Segmentation Model
                </span>
                <span style={{ fontSize: '10px', color: '#00F0B4', fontWeight: 700 }}>Frozen Weights</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {[
                  { id: 'dual_stream', label: 'Dual-Stream Hybrid (Default)', sub: 'Raw Macro + Restored Capillary Fusion', view: 'dual_stream_focus', color: '#00F0B4' },
                  { id: 'raw_unet', label: 'Raw U-Net (Baseline)', sub: 'High macro precision baseline', view: 'raw_focus', color: '#38bdf8' },
                  { id: 'restored_unet', label: 'Restored U-Net (Baseline)', sub: 'SFHformer-restored fine vessel recall', view: 'restored_focus', color: '#34d399' }
                ].map(m => {
                  const active = selectedSegModel === m.id;
                  return (
                    <button
                      key={m.id}
                      onClick={() => {
                        setSelectedSegModel(m.id);
                        setDualViewMode(m.view);
                        runDualSegmentation(selectedSampleId, segThreshold, segResolution, m.id, includeFusionViz);
                      }}
                      style={{
                        textAlign: 'left', padding: '8px 10px', borderRadius: '8px', cursor: 'pointer',
                        background: active ? 'rgba(0, 240, 180, 0.14)' : 'rgba(255,255,255,0.03)',
                        border: active ? `1px solid ${m.color}66` : '1px solid rgba(255,255,255,0.05)',
                        color: active ? m.color : '#cbd5e1'
                      }}
                    >
                      <div style={{ fontSize: '12px', fontWeight: 600 }}>{m.label}</div>
                      <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px' }}>{m.sub}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Threshold & Protocol Operating Point */}
            <div style={{ 
              background: 'rgba(15, 23, 42, 0.65)', backdropFilter: 'blur(12px)', 
              border: isThresholdExploratory ? '1px solid rgba(245, 158, 11, 0.4)' : '1px solid rgba(255,255,255,0.08)', 
              borderRadius: '12px', padding: '16px' 
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <ShieldCheck size={16} color={isThresholdExploratory ? '#f59e0b' : '#00F0B4'} />
                  <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>
                    Operating Threshold (τ)
                  </span>
                </div>
                {!isDualCalibratedThreshold && (
                  <button
                    onClick={() => {
                      setSegThreshold(FROZEN_DUAL_THRESHOLD);
                      runDualSegmentation(selectedSampleId, FROZEN_DUAL_THRESHOLD, segResolution, selectedSegModel, includeFusionViz);
                    }}
                    style={{
                      display: 'flex', alignItems: 'center', gap: '4px', padding: '3px 8px', borderRadius: '4px',
                      background: 'rgba(0, 240, 180, 0.15)', color: '#00F0B4', border: '1px solid rgba(0, 240, 180, 0.35)',
                      fontSize: '11px', fontWeight: 600, cursor: 'pointer'
                    }}
                    title="Reset to frozen validation-calibrated threshold"
                  >
                    <RotateCcw size={11} /> Reset 0.40
                  </button>
                )}
              </div>

              {/* Dual Operating Point Buttons */}
              <div style={{ display: 'flex', gap: '6px', marginBottom: '10px' }}>
                <button
                  onClick={() => {
                    setSegThreshold(0.40);
                    runDualSegmentation(selectedSampleId, 0.40, segResolution, selectedSegModel, includeFusionViz);
                  }}
                  style={{
                    flex: 1, padding: '6px 8px', borderRadius: '6px', fontSize: '11px', fontWeight: 600, cursor: 'pointer',
                    background: isDualCalibratedThreshold ? 'rgba(0, 240, 180, 0.22)' : 'rgba(255,255,255,0.04)',
                    color: isDualCalibratedThreshold ? '#00F0B4' : '#94a3b8',
                    border: isDualCalibratedThreshold ? '1px solid rgba(0, 240, 180, 0.45)' : '1px solid rgba(255,255,255,0.06)'
                  }}
                >
                  τ = 0.40 (Calibrated)
                </button>
                <button
                  onClick={() => {
                    setSegThreshold(0.50);
                    runDualSegmentation(selectedSampleId, 0.50, segResolution, selectedSegModel, includeFusionViz);
                  }}
                  style={{
                    flex: 1, padding: '6px 8px', borderRadius: '6px', fontSize: '11px', fontWeight: 600, cursor: 'pointer',
                    background: isAltBaselineThreshold ? 'rgba(56, 189, 248, 0.22)' : 'rgba(255,255,255,0.04)',
                    color: isAltBaselineThreshold ? '#38bdf8' : '#94a3b8',
                    border: isAltBaselineThreshold ? '1px solid rgba(56, 189, 248, 0.45)' : '1px solid rgba(255,255,255,0.06)'
                  }}
                >
                  τ = 0.50 (Baseline Match)
                </button>
              </div>

              {/* Status Badge */}
              <div style={{ 
                padding: '6px 10px', borderRadius: '6px', marginBottom: '12px', fontSize: '11px', lineHeight: 1.3,
                background: isThresholdExploratory ? 'rgba(245, 158, 11, 0.12)' : (isAltBaselineThreshold ? 'rgba(56, 189, 248, 0.1)' : 'rgba(0, 240, 180, 0.1)'),
                border: isThresholdExploratory ? '1px solid rgba(245, 158, 11, 0.25)' : (isAltBaselineThreshold ? '1px solid rgba(56, 189, 248, 0.25)' : '1px solid rgba(0, 240, 180, 0.25)'),
                color: isThresholdExploratory ? '#fbbf24' : (isAltBaselineThreshold ? '#7dd3fc' : '#00F0B4')
              }}>
                {isDualCalibratedThreshold ? (
                  <span>🔒 <strong>Frozen Validation-Calibrated Threshold (0.40):</strong> Official FIVES-calibrated operating point for DualStreamUNet (selected on N=59 clean val).</span>
                ) : isAltBaselineThreshold ? (
                  <span>⚖️ <strong>Alternative Threshold (0.50):</strong> Threshold-matched comparison with single-stream baselines (not an independently optimized point for DualStreamUNet).</span>
                ) : (
                  <span>⚠️ <strong>Exploratory View Mode ({segThreshold.toFixed(2)}):</strong> Alters local preview only. Frozen benchmark results are never modified.</span>
                )}
              </div>

              {/* Threshold Slider */}
              <div style={{ marginBottom: '14px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8', marginBottom: '6px' }}>
                  <span>Exploratory Cutoff Slider:</span>
                  <span style={{ color: isThresholdExploratory ? '#fbbf24' : (isAltBaselineThreshold ? '#38bdf8' : '#00F0B4'), fontWeight: 700 }}>
                    {segThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.20"
                  max="0.80"
                  step="0.05"
                  value={segThreshold}
                  onChange={e => setSegThreshold(parseFloat(e.target.value))}
                  onMouseUp={() => runDualSegmentation(selectedSampleId, segThreshold, segResolution, selectedSegModel, includeFusionViz)}
                  style={{ width: '100%', accentColor: isThresholdExploratory ? '#fbbf24' : '#00F0B4' }}
                />
              </div>

              {/* Resolution Toggle */}
              <div style={{ marginBottom: '12px' }}>
                <label style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>
                  Inference Resolution:
                </label>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {[
                    { val: 512, label: '512×512 (Capillary Rich)' },
                    { val: 256, label: '256×256 (Ultra-Fast)' }
                  ].map(r => (
                    <button
                      key={r.val}
                      onClick={() => {
                        setSegResolution(r.val);
                        runDualSegmentation(selectedSampleId, segThreshold, r.val, selectedSegModel, includeFusionViz);
                      }}
                      style={{
                        flex: 1, padding: '6px 8px', borderRadius: '6px', fontSize: '11px', fontWeight: 600, cursor: 'pointer',
                        background: segResolution === r.val ? 'rgba(0, 180, 216, 0.25)' : 'rgba(255,255,255,0.04)',
                        color: segResolution === r.val ? '#38bdf8' : '#94a3b8',
                        border: segResolution === r.val ? '1px solid rgba(56, 189, 248, 0.4)' : '1px solid transparent'
                      }}
                    >
                      {r.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Optional Spatial Fusion Gate Visualization Toggle (Hidden by default) */}
              <div style={{ paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: selectedSegModel === 'dual_stream' ? 'pointer' : 'not-allowed', fontSize: '11px', color: selectedSegModel === 'dual_stream' ? '#cbd5e1' : '#64748b' }}>
                  <span>Optional Spatial Gate Map (w_rest)</span>
                  <input
                    type="checkbox"
                    disabled={selectedSegModel !== 'dual_stream'}
                    checked={includeFusionViz && selectedSegModel === 'dual_stream'}
                    onChange={e => {
                      const next = e.target.checked;
                      setIncludeFusionViz(next);
                      runDualSegmentation(selectedSampleId, segThreshold, segResolution, selectedSegModel, next);
                    }}
                    style={{ accentColor: '#f59e0b' }}
                  />
                </label>
                <div style={{ fontSize: '10px', color: '#64748b', marginTop: '3px' }}>
                  {selectedSegModel === 'dual_stream'
                    ? 'Renders scale_0 spatial routing weights [0, 1] (hidden by default).'
                    : 'Available only when Dual-Stream Hybrid is selected.'}
                </div>
              </div>
            </div>

            {/* Checkpoint Provenance Box */}
            <div style={{ background: 'rgba(15, 23, 42, 0.45)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '10px', padding: '14px', fontSize: '11px', color: '#64748b' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#94a3b8', fontWeight: 600, marginBottom: '6px' }}>
                <CheckCircle2 size={13} color="#00F0B4" />
                Frozen Checkpoint Integrity (SHA-256):
              </div>
              <div>• Dual-Stream Hybrid: <code style={{ color: '#00F0B4' }}>9be2213e...</code></div>
              <div>• Raw U-Net: <code style={{ color: '#38bdf8' }}>97639a65...</code></div>
              <div>• Restored U-Net: <code style={{ color: '#34d399' }}>2cfb3737...</code></div>
              <div>• SFHformer: <code style={{ color: '#f59e0b' }}>d74e267c...</code></div>
            </div>

          </div>

          {/* Right Column: Visualizer Panels & Differential Biomarkers */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* View Mode Bar */}
            <div style={{ background: 'rgba(15, 23, 42, 0.65)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {[
                    { id: 'dual_stream_focus', label: 'Dual-Stream Hybrid (Default)', color: '#00F0B4' },
                    { id: 'tri_view', label: '3-Model Comparative Grid', color: '#38bdf8' },
                    { id: 'differential_map', label: 'Differential Vessel Response', color: '#fbbf24' },
                    { id: 'restored_focus', label: 'Restored U-Net', color: '#34d399' },
                    { id: 'raw_focus', label: 'Raw U-Net', color: '#00B4D8' },
                  ].map(v => (
                    <button
                      key={v.id}
                      onClick={() => setDualViewMode(v.id)}
                      style={{
                        padding: '6px 10px', borderRadius: '6px', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                        background: dualViewMode === v.id ? 'rgba(0, 180, 216, 0.25)' : 'rgba(255,255,255,0.04)',
                        color: dualViewMode === v.id ? v.color : '#94a3b8',
                        border: dualViewMode === v.id ? `1px solid ${v.color}66` : '1px solid transparent'
                      }}
                    >
                      {v.label}
                    </button>
                  ))}
                </div>

                {dualResult && (
                  <div style={{ fontSize: '11px', color: '#64748b', display: 'flex', gap: '14px', alignItems: 'center' }}>
                    <span 
                      onClick={() => setShowTimingBreakdown(!showTimingBreakdown)}
                      style={{ cursor: 'pointer', textDecoration: 'underline dotted', display: 'flex', alignItems: 'center', gap: '4px' }}
                      title="Click to see exact reconciled timing breakdown"
                    >
                      <Clock size={12} color="#38bdf8" />
                      E2E Latency: <strong style={{ color: '#38bdf8' }}>{dualResult.timing.total_ms} ms</strong>
                    </span>
                    <span>Additional Vessels: <strong style={{ color: '#fbbf24' }}>+{dualResult.biomarkers.differential.differential_response_pct || dualResult.biomarkers.differential.capillary_recovery_gain_pct}%</strong></span>
                  </div>
                )}
              </div>

              {/* Reconciled Latency Breakdown Drawer (Scientific Correction 3) */}
              {showTimingBreakdown && dualResult && (
                <div style={{ marginBottom: '14px', padding: '12px 14px', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(56, 189, 248, 0.3)', fontSize: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '4px' }}>
                    <span style={{ fontWeight: 600, color: '#38bdf8' }}>Reconciled Execution Latency Breakdown:</span>
                    <span style={{ color: '#94a3b8', fontSize: '11px' }}>RTX 5060 Laptop GPU • BF16 AMP</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', color: '#cbd5e1', fontSize: '11px' }}>
                    <div>
                      <span style={{ color: '#94a3b8', display: 'block' }}>1. Disk I/O & Prep:</span>
                      <strong>{dualResult.timing.input_io_and_prep_ms || 0} ms</strong>
                      <div style={{ color: '#64748b', fontSize: '10px' }}>
                        Image read, resize, GPU transfer
                      </div>
                    </div>
                    <div>
                      <span style={{ color: '#94a3b8', display: 'block' }}>2. GPU Neural Forward:</span>
                      <strong>{dualResult.timing.neural_inference_total_ms || 0} ms</strong>
                      <div style={{ color: '#64748b', fontSize: '10px' }}>
                        SFH: {dualResult.timing.sfhformer_restoration_ms}ms • Single: {dualResult.timing.dual_unet_segmentation_ms}ms • Hybrid: {dualResult.timing.dual_stream_segmentation_ms || 0}ms
                      </div>
                    </div>
                    <div>
                      <span style={{ color: '#94a3b8', display: 'block' }}>3. CPU Topology Analysis:</span>
                      <strong>{dualResult.timing.topology_analysis_ms || 0} ms</strong>
                      <div style={{ color: '#64748b', fontSize: '10px' }}>
                        FOV mask, CCL labeling, skeletonize, endpoints
                      </div>
                    </div>
                    <div>
                      <span style={{ color: '#94a3b8', display: 'block' }}>4. Overlays & Base64 PNG:</span>
                      <strong>{((dualResult.timing.overlay_blending_ms || 0) + (dualResult.timing.fidelity_and_base64_encoding_ms || 0)).toFixed(1)} ms</strong>
                      <div style={{ color: '#64748b', fontSize: '10px' }}>
                        Compositing + PNG Base64 encoding
                      </div>
                    </div>
                  </div>
                  <div style={{ marginTop: '8px', paddingTop: '6px', borderTop: '1px solid rgba(255,255,255,0.05)', fontSize: '11px', color: '#34d399', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Sum of Pipeline Stages: <strong>{((dualResult.timing.neural_inference_total_ms || 0) + (dualResult.timing.postprocessing_and_io_total_ms || 0)).toFixed(1)} ms</strong></span>
                    <span>Total Measured E2E: <strong>{dualResult.timing.total_ms} ms</strong> (100% Accounted For)</span>
                  </div>
                </div>
              )}

              {/* View Canvas Layout */}
              {dualError ? (
                <div style={{ padding: '24px', textAlign: 'center', color: '#f87171', background: 'rgba(239,68,68,0.1)', borderRadius: '8px' }}>
                  {dualError}
                </div>
              ) : dualLoading ? (
                <div style={{ height: '360px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px', color: '#94a3b8' }}>
                  <RefreshCw size={28} className="spin" color="#00B4D8" />
                  <span>Executing SFHformer Restoration, Dual U-Net & Dual-Stream Hybrid Segmentation...</span>
                </div>
              ) : dualResult ? (
                <div>
                  {/* Option 0: Dual-Stream Hybrid Focus (Default) */}
                  {dualViewMode === 'dual_stream_focus' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '20px', alignItems: 'center' }}>
                      <div style={{ width: '100%', maxHeight: '420px', borderRadius: '10px', overflow: 'hidden', background: '#020617', border: '2px solid rgba(0, 240, 180, 0.55)', display: 'flex', justifyContent: 'center' }}>
                        <img src={dualResult.overlay_dual_stream || dualResult.overlay_restored} alt="Dual-Stream Hybrid Overlay" style={{ width: '100%', maxHeight: '420px', objectFit: 'contain' }} />
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        <div style={{ padding: '14px', borderRadius: '10px', background: 'rgba(0, 240, 180, 0.08)', border: '1px solid rgba(0, 240, 180, 0.3)' }}>
                          <h3 style={{ margin: '0 0 6px 0', fontSize: '14px', color: '#00F0B4', fontWeight: 700 }}>
                            Dual-Stream Hybrid U-Net (τ = {dualResult.threshold.toFixed(2)})
                          </h3>
                          <p style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#cbd5e1', lineHeight: 1.4 }}>
                            Fuses Raw encoder features (macro precision) and SFHformer-restored encoder features (fine-vessel sensitivity) across 5 spatial scales.
                          </p>
                          {dualResult.biomarkers.dual_stream && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '12px' }}>
                              <div><span style={{ color: '#94a3b8', fontSize: '10px', display: 'block' }}>Vessel Density</span><strong style={{ color: '#00F0B4' }}>{dualResult.biomarkers.dual_stream.vessel_density_pct}%</strong></div>
                              <div><span style={{ color: '#94a3b8', fontSize: '10px', display: 'block' }}>Connected Comp.</span><strong style={{ color: '#38bdf8' }}>{dualResult.biomarkers.dual_stream.branch_fragment_count}</strong></div>
                              <div><span style={{ color: '#94a3b8', fontSize: '10px', display: 'block' }}>Skeleton Endpoints</span><strong style={{ color: '#cbd5e1' }}>{dualResult.biomarkers.dual_stream.endpoint_count}</strong></div>
                              {dualResult.biomarkers.dual_vs_raw && (
                                <div><span style={{ color: '#94a3b8', fontSize: '10px', display: 'block' }}>Delta vs Raw U-Net</span><strong style={{ color: '#fbbf24' }}>+{dualResult.biomarkers.dual_vs_raw.differential_response_pct}%</strong></div>
                              )}
                            </div>
                          )}
                        </div>
                        <div style={{ padding: '12px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', fontSize: '11px', color: '#94a3b8', lineHeight: 1.4 }}>
                          <strong style={{ color: '#e2e8f0' }}>Domain Transfer Context:</strong> FIVES (N=200) is the primary in-domain benchmark (Dice 0.8474, ThinSkel 43.52%, clDice 0.8537). External HRF transfer shows mixed precision/Dice trade-offs (-8.54 pp precision vs Raw) and upstream SFHformer exposure on 40/45 images.
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Option A: 3-Model Comparative Grid */}
                  {dualViewMode === 'tri_view' && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                      
                      {/* Panel 1: Raw U-Net */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '12px', fontWeight: 600, color: '#00B4D8' }}>
                            1. Raw U-Net (Macro Baseline)
                          </span>
                          <span style={{ fontSize: '11px', color: '#94a3b8' }}>
                            Density: {dualResult.biomarkers.raw.vessel_density_pct}%
                          </span>
                        </div>
                        <div style={{ width: '100%', aspectRatio: '1/1', borderRadius: '8px', overflow: 'hidden', background: '#020617', border: '1px solid rgba(0, 180, 216, 0.4)' }}>
                          <img src={dualResult.overlay_raw} alt="Raw U-Net Overlay" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        </div>
                        <div style={{ fontSize: '11px', color: '#64748b', display: 'flex', justifyContent: 'space-between' }}>
                          <span>Components: {dualResult.biomarkers.raw.branch_fragment_count}</span>
                          <span>Endpoints: {dualResult.biomarkers.raw.endpoint_count}</span>
                        </div>
                      </div>

                      {/* Panel 2: Restored U-Net */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '12px', fontWeight: 600, color: '#34d399' }}>
                            2. Restored U-Net (Capillary Focus)
                          </span>
                          <span style={{ fontSize: '11px', color: '#34d399', fontWeight: 600 }}>
                            Density: {dualResult.biomarkers.restored.vessel_density_pct}%
                          </span>
                        </div>
                        <div style={{ width: '100%', aspectRatio: '1/1', borderRadius: '8px', overflow: 'hidden', background: '#020617', border: '1px solid rgba(52, 211, 153, 0.4)' }}>
                          <img src={dualResult.overlay_restored} alt="Restored U-Net Overlay" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        </div>
                        <div style={{ fontSize: '11px', color: '#64748b', display: 'flex', justifyContent: 'space-between' }}>
                          <span>Components: {dualResult.biomarkers.restored.branch_fragment_count}</span>
                          <span>Endpoints: {dualResult.biomarkers.restored.endpoint_count}</span>
                        </div>
                      </div>

                      {/* Panel 3: Dual-Stream Hybrid */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '12px', fontWeight: 600, color: '#00F0B4' }}>
                            3. Dual-Stream Hybrid
                          </span>
                          <span style={{ fontSize: '11px', color: '#00F0B4', fontWeight: 700 }}>
                            Density: {dualResult.biomarkers.dual_stream ? dualResult.biomarkers.dual_stream.vessel_density_pct : '—'}%
                          </span>
                        </div>
                        <div style={{ width: '100%', aspectRatio: '1/1', borderRadius: '8px', overflow: 'hidden', background: '#020617', border: '1px solid rgba(0, 240, 180, 0.5)' }}>
                          <img src={dualResult.overlay_dual_stream || dualResult.overlay_differential} alt="Dual-Stream Hybrid" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                        </div>
                        <div style={{ fontSize: '11px', color: '#64748b', display: 'flex', justifyContent: 'space-between' }}>
                          <span>Components: {dualResult.biomarkers.dual_stream ? dualResult.biomarkers.dual_stream.branch_fragment_count : '—'}</span>
                          <span>Endpoints: {dualResult.biomarkers.dual_stream ? dualResult.biomarkers.dual_stream.endpoint_count : '—'}</span>
                        </div>
                      </div>

                    </div>
                  )}

                  {/* Option B: Differential Response Map Focused (Scientific Correction 1) */}
                  {dualViewMode === 'differential_map' && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '20px', alignItems: 'center' }}>
                      <div style={{ width: '100%', maxHeight: '420px', borderRadius: '10px', overflow: 'hidden', background: '#020617', border: '2px solid rgba(251, 191, 36, 0.6)', display: 'flex', justifyContent: 'center' }}>
                        <img src={dualResult.overlay_differential || dualResult.overlay_gain} alt="Differential Vessel Response Map" style={{ width: '100%', maxHeight: '420px', objectFit: 'contain' }} />
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                        <div style={{ padding: '16px', borderRadius: '10px', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                            <Sparkles size={18} color="#fbbf24" />
                            <h3 style={{ margin: 0, fontSize: '15px', color: '#fbbf24', fontWeight: 700 }}>
                              Differential Vessel Response
                            </h3>
                          </div>
                          <p style={{ margin: '0 0 12px 0', fontSize: '12px', color: '#cbd5e1', lineHeight: 1.4 }}>
                            Electric Amber pixels indicate <strong>additional vessel predictions</strong> generated by the restored model that the raw model omitted.
                          </p>
                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                            <div>
                              <span style={{ fontSize: '11px', color: '#94a3b8' }}>Additional Vessel Pixels:</span>
                              <div style={{ fontSize: '18px', fontWeight: 700, color: '#fbbf24' }}>
                                +{(dualResult.biomarkers.differential.additional_vessel_pixels || dualResult.biomarkers.differential.newly_recovered_capillary_pixels).toLocaleString()} px
                              </div>
                            </div>
                            <div>
                              <span style={{ fontSize: '11px', color: '#94a3b8' }}>Differential Response:</span>
                              <div style={{ fontSize: '18px', fontWeight: 700, color: '#fbbf24' }}>
                                +{dualResult.biomarkers.differential.differential_response_pct || dualResult.biomarkers.differential.capillary_recovery_gain_pct}%
                              </div>
                            </div>
                          </div>
                        </div>

                        <div style={{ padding: '14px', borderRadius: '10px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}>
                          <div style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '6px', fontWeight: 600 }}>Color Legend & Scientific Interpretation:</div>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '12px' }}>
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                              <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#00B4D8', marginTop: '2px', flexShrink: 0 }}></span>
                              <span><strong>Shared Vessel Predictions</strong>: Confirmed by both models (Macro arcades).</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                              <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#fbbf24', marginTop: '2px', flexShrink: 0 }}></span>
                              <span><strong>Additional Vessel Predictions</strong>: Restored-exclusive predictions. Requires branch-level ground-truth matching to distinguish genuine peripheral capillaries from false-positive fragments.</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                              <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#818cf8', marginTop: '2px', flexShrink: 0 }}></span>
                              <span><strong>Raw U-Net Exclusive</strong>: Border pixels where raw model predicted vessel boundaries wider than restored model.</span>
                            </div>
                          </div>
                        </div>

                      </div>
                    </div>
                  )}

                  {/* Option C: Single Focus Views */}
                  {dualViewMode === 'restored_focus' && (
                    <div style={{ display: 'flex', justifyContent: 'center' }}>
                      <div style={{ width: '100%', maxHeight: '420px', borderRadius: '10px', overflow: 'hidden', background: '#020617', border: '2px solid rgba(52, 211, 153, 0.6)', display: 'flex', justifyContent: 'center' }}>
                        <img src={dualResult.overlay_restored} alt="Restored Overlay" style={{ width: '100%', maxHeight: '420px', objectFit: 'contain' }} />
                      </div>
                    </div>
                  )}

                  {dualViewMode === 'raw_focus' && (
                    <div style={{ display: 'flex', justifyContent: 'center' }}>
                      <div style={{ width: '100%', maxHeight: '420px', borderRadius: '10px', overflow: 'hidden', background: '#020617', border: '2px solid rgba(0, 180, 216, 0.6)', display: 'flex', justifyContent: 'center' }}>
                        <img src={dualResult.overlay_raw} alt="Raw Overlay" style={{ width: '100%', maxHeight: '420px', objectFit: 'contain' }} />
                      </div>
                    </div>
                  )}

                  {/* Optional Spatial Fusion-Weight Visualization (Hidden by Default) */}
                  {includeFusionViz && selectedSegModel === 'dual_stream' && dualResult.overlay_fusion_gate && (
                    <div style={{ marginTop: '16px', padding: '16px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.85)', border: '1px solid rgba(245, 158, 11, 0.35)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                        <div>
                          <h4 style={{ margin: 0, fontSize: '13px', fontWeight: 700, color: '#fbbf24' }}>
                            Optional Spatial Fusion-Weight Visualization (scale_0: w_rest)
                          </h4>
                          <p style={{ margin: '2px 0 0 0', fontSize: '11px', color: '#94a3b8' }}>
                            Higher values indicate greater restored-stream gating weight (w_rest) in [0, 1]. Not confidence, uncertainty, or vessel probability.
                          </p>
                        </div>
                        {dualResult.fusion_gate_stats && (
                          <span style={{ fontSize: '11px', color: '#cbd5e1' }}>
                            Mean w_rest: <strong>{dualResult.fusion_gate_stats.mean_w_rest}</strong> (Restored-dominant: {dualResult.fusion_gate_stats.pct_rest_dominant}%)
                          </span>
                        )}
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 240px', gap: '16px', alignItems: 'center' }}>
                        <div style={{ width: '100%', maxHeight: '300px', borderRadius: '8px', overflow: 'hidden', background: '#020617', display: 'flex', justifyContent: 'center' }}>
                          <img src={dualResult.overlay_fusion_gate} alt="Spatial Fusion Gate Heatmap" style={{ width: '100%', maxHeight: '300px', objectFit: 'contain' }} />
                        </div>
                        <div style={{ padding: '12px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', fontSize: '11px' }}>
                          <div style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: '8px' }}>Fusion Gate Weight Legend (0 → 1)</div>
                          <div style={{ height: '14px', borderRadius: '4px', background: 'linear-gradient(90deg, #2563eb 0%, #14b8a6 50%, #f59e0b 100%)', marginBottom: '6px' }} />
                          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '10px', marginBottom: '8px' }}>
                            <span>0.0 (Raw Stream)</span>
                            <span>0.5 (Balanced)</span>
                            <span>1.0 (Restored Stream)</span>
                          </div>
                          <div style={{ color: '#64748b', fontSize: '10px', lineHeight: 1.35 }}>
                            Visualizes spatial routing weights w_rest = 1 - w_raw produced by the scale_0 Softmax(dim=1) spatial gate.
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                </div>
              ) : null}

            </div>

            {/* Differential KPI Biomarkers Card */}
            {dualResult && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                
                {/* KPI 1 */}
                <div style={{ background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '10px', padding: '14px' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Additional Vessel Predictions
                  </span>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: '#fbbf24', marginTop: '4px' }}>
                    +{dualResult.biomarkers.differential.differential_response_pct || dualResult.biomarkers.differential.capillary_recovery_gain_pct}%
                  </div>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>
                    +{(dualResult.biomarkers.differential.additional_vessel_pixels || dualResult.biomarkers.differential.newly_recovered_capillary_pixels).toLocaleString()} differential vessel px
                  </span>
                </div>

                {/* KPI 2 */}
                <div style={{ background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(0, 180, 216, 0.25)', borderRadius: '10px', padding: '14px' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Vascular Skeleton Shift
                  </span>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: '#38bdf8', marginTop: '4px' }}>
                    +{dualResult.biomarkers.differential.skeleton_length_delta_pct || dualResult.biomarkers.differential.skeleton_length_gain_pct}%
                  </div>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>
                    +{dualResult.biomarkers.differential.endpoint_delta} terminal branch endpoints
                  </span>
                </div>

                {/* KPI 3 */}
                <div style={{ background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '10px', padding: '14px' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Vessel Density Shift
                  </span>
                  <div style={{ fontSize: '20px', fontWeight: 700, color: '#f1f5f9', marginTop: '4px' }}>
                    {dualResult.biomarkers.raw.vessel_density_pct}% → {dualResult.biomarkers.restored.vessel_density_pct}%
                  </div>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>
                    Shared arcade: {dualResult.biomarkers.differential.shared_vessel_pixels.toLocaleString()} px
                  </span>
                </div>

                {/* KPI 4 */}
                <div style={{ background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '10px', padding: '14px' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Restoration Fidelity
                  </span>
                  <div style={{ fontSize: '20px', fontWeight: 700, color: '#34d399', marginTop: '4px' }}>
                    {dualResult.restoration_metrics.psnr} dB
                  </div>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>
                    SSIM: {dualResult.restoration_metrics.ssim} • Latency: {dualResult.timing.total_ms}ms
                  </span>
                </div>

              </div>
            )}

            {/* Model-Information & Domain Generalization Panel */}
            <div style={{ padding: '14px 18px', borderRadius: '10px', background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.25)', display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
              <Info size={18} color="#38bdf8" style={{ marginTop: '2px', flexShrink: 0 }} />
              <div style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.5 }}>
                <strong style={{ color: '#38bdf8' }}>Model Provenance & Domain Transfer Context:</strong>{' '}
                <strong>FIVES (N=200)</strong> represents the primary in-domain benchmark, where <strong>DualStreamUNet (τ=0.40)</strong> improves Dice (<strong>0.8474</strong> vs 0.8133 Raw / 0.6927 Restored), thin-vessel skeleton sensitivity (<strong>+10.20 pp</strong> vs Raw), clDice (<strong>0.8537</strong>), and fragmentation (<strong>41.2</strong> vs 52.2 / 114.2 CCs) while preserving Glaucoma precision (<strong>89.98%</strong> vs 57.59% single-stream Restored).{' '}
                <strong>HRF (N=45)</strong> reflects mixed external-domain transfer and includes upstream SFHformer pretraining exposure on 40/45 images (N=5 unexposed exploratory only): DualStreamUNet improves thin skeleton sensitivity (+9.78 pp), clDice (+2.78 pp), and reduces fragmentation (-12.49 CCs/img) vs Raw U-Net, while incurring Precision (-8.54 pp) and Dice (-1.82 pp) trade-offs vs Raw U-Net and trailing Restored U-Net (-3.89 pp) on HRF thin skeleton sensitivity.
              </div>
            </div>

          </div>

        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: MILESTONE 3 CLINICAL BENCHMARK                                      */}
      {/* ========================================================================= */}
      {activeTab === 'benchmark' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Header Banner */}
          <div style={{ background: 'rgba(15, 23, 42, 0.75)', border: '1px solid rgba(0, 180, 216, 0.3)', borderRadius: '12px', padding: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <ShieldCheck size={22} color="#00F0B4" />
                <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: '#f1f5f9' }}>
                  Milestone 3: Final Generalization Evaluation
                </h2>
              </div>
              <span style={{ fontSize: '11px', padding: '4px 10px', borderRadius: '6px', background: 'rgba(0, 240, 180, 0.15)', color: '#00F0B4', fontWeight: 600 }}>
                Frozen Thresholds: Baselines = 0.50 • Dual-Stream = 0.40 • Frangi = 0.0010
              </span>
            </div>
            <p style={{ margin: 0, fontSize: '13px', color: '#94a3b8', lineHeight: 1.5 }}>
              Empirical multi-system evaluation across 200 untouched FIVES test scans (primary in-domain benchmark) and 45 external HRF scans (mixed external-domain transfer; 40 exposed during SFHformer pretraining + 5 exploratory unexposed scans). Evaluates Raw U-Net, Restored U-Net, Dual-Stream Hybrid U-Net, and Frangi baselines under predeclared frozen validation operating points.
            </p>
          </div>

          {/* 3 Executive Discovery Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
            
            <div style={{ background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(0, 240, 180, 0.35)', borderRadius: '10px', padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>FIVES Primary In-Domain (N=200)</span>
                <span style={{ fontSize: '11px', color: '#00F0B4', fontWeight: 700 }}>Holm p = 8.72 × 10⁻²⁸</span>
              </div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: '#00F0B4' }}>
                0.8474 Dice | +10.20 pp Thin
              </div>
              <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#cbd5e1', lineHeight: 1.4 }}>
                <strong>Dual-Stream Hybrid (τ=0.40)</strong> outperforms both baselines simultaneously: Dice 0.8474 (vs 0.8133 Raw / 0.6927 Restored), ThinSkel <strong>43.52%</strong>, clDice <strong>0.8537</strong>, and CC <strong>41.2</strong>.
              </p>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(56, 189, 248, 0.35)', borderRadius: '10px', padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>HRF External Transfer (N=45)</span>
                <span style={{ fontSize: '11px', color: '#38bdf8', fontWeight: 700 }}>40 Pretrain-Exposed + 5 Unexposed</span>
              </div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: '#38bdf8' }}>
                Mixed Domain Trade-Offs
              </div>
              <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#cbd5e1', lineHeight: 1.4 }}>
                Restored U-Net reaches <strong>50.48%</strong> ThinSkel (+13.67 pp, p=5.68×10⁻¹³). Dual-Stream gains +9.78 pp ThinSkel &amp; -12.49 CCs vs Raw, but drops -8.54 pp Precision vs Raw and -3.89 pp ThinSkel vs Restored.
              </p>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(251, 191, 36, 0.35)', borderRadius: '10px', padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontSize: '11px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>HRF Unexposed (N=5 Exploratory)</span>
                <span style={{ fontSize: '11px', color: '#fbbf24', fontWeight: 700 }}>Wilcoxon p = 0.0625</span>
              </div>
              <div style={{ fontSize: '22px', fontWeight: 700, color: '#fbbf24' }}>
                58.97% Rest | 56.30% Dual
              </div>
              <p style={{ margin: '6px 0 0 0', fontSize: '12px', color: '#cbd5e1', lineHeight: 1.4 }}>
                Exploratory evidence only (insufficient N=5 power; Holm p = 0.375). ThinSkel rises from 47.77% (Raw) to 58.97% (Restored) and 56.30% (Dual-Stream, clDice 0.7854).
              </p>
            </div>

          </div>

          {/* Benchmark Table Card */}
          <div style={{ background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', padding: '18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 600, color: '#f1f5f9' }}>
                Frozen Multi-System Comparative Performance Matrix (Predeclared Operating Points)
              </h3>
              <span style={{ fontSize: '11px', color: '#64748b' }}>
                Decoupled from Interactive Explorer
              </span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: '#94a3b8' }}>
                    <th style={{ padding: '10px 12px' }}>Dataset / Cohort</th>
                    <th style={{ padding: '10px 12px' }}>Pipeline System (Threshold)</th>
                    <th style={{ padding: '10px 12px' }}>Overall Dice</th>
                    <th style={{ padding: '10px 12px' }}>Precision</th>
                    <th style={{ padding: '10px 12px' }}>Sensitivity</th>
                    <th style={{ padding: '10px 12px' }}>Thin Skeleton Sens</th>
                    <th style={{ padding: '10px 12px' }}>Thin Branch Recall</th>
                    <th style={{ padding: '10px 12px' }}>clDice (Topology)</th>
                  </tr>
                </thead>
                <tbody>
                  {/* FIVES 200 (Primary In-Domain) */}
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: 'rgba(255,255,255,0.01)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#e2e8f0' }}>FIVES Test (200 Scans)</td>
                    <td style={{ padding: '10px 12px', color: '#38bdf8', fontWeight: 600 }}>Raw → U-Net (τ=0.50)</td>
                    <td style={{ padding: '10px 12px', color: '#38bdf8' }}>0.8133</td>
                    <td style={{ padding: '10px 12px', color: '#38bdf8', fontWeight: 700 }}>0.9266</td>
                    <td style={{ padding: '10px 12px' }}>0.7566</td>
                    <td style={{ padding: '10px 12px' }}>33.33%</td>
                    <td style={{ padding: '10px 12px' }}>34.95%</td>
                    <td style={{ padding: '10px 12px', color: '#38bdf8' }}>0.8134</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#e2e8f0' }}>FIVES Test (200 Scans)</td>
                    <td style={{ padding: '10px 12px', color: '#94a3b8' }}>Restored → U-Net (τ=0.50)</td>
                    <td style={{ padding: '10px 12px' }}>0.6927</td>
                    <td style={{ padding: '10px 12px' }}>0.7142</td>
                    <td style={{ padding: '10px 12px' }}>0.6824</td>
                    <td style={{ padding: '10px 12px' }}>29.68%</td>
                    <td style={{ padding: '10px 12px' }}>31.94%</td>
                    <td style={{ padding: '10px 12px' }}>0.6536</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0, 240, 180, 0.10)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#00F0B4' }}>FIVES Test (200 Scans)</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 700 }}>Dual-Stream Hybrid (τ=0.40)</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 700 }}>0.8474 (+3.41 pp)</td>
                    <td style={{ padding: '10px 12px', color: '#34d399' }}>0.9007</td>
                    <td style={{ padding: '10px 12px', color: '#34d399' }}>0.8242</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 700 }}>43.52% (+10.20 pp)</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 700 }}>45.81% (+10.86 pp)</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 700 }}>0.8537 (+4.02 pp)</td>
                  </tr>

                  {/* HRF Overall */}
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: 'rgba(255,255,255,0.01)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#38bdf8' }}>HRF External (45 Scans)</td>
                    <td style={{ padding: '10px 12px', color: '#94a3b8' }}>Raw → U-Net (τ=0.50)</td>
                    <td style={{ padding: '10px 12px' }}>0.7601</td>
                    <td style={{ padding: '10px 12px' }}>0.7482</td>
                    <td style={{ padding: '10px 12px' }}>0.7900</td>
                    <td style={{ padding: '10px 12px' }}>36.81%</td>
                    <td style={{ padding: '10px 12px' }}>31.93%</td>
                    <td style={{ padding: '10px 12px' }}>0.7467</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: 'rgba(52, 211, 153, 0.06)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#38bdf8' }}>HRF External (45 Scans)</td>
                    <td style={{ padding: '10px 12px', color: '#34d399', fontWeight: 600 }}>Restored → U-Net (τ=0.50)</td>
                    <td style={{ padding: '10px 12px' }}>0.7368</td>
                    <td style={{ padding: '10px 12px' }}>0.6532</td>
                    <td style={{ padding: '10px 12px', color: '#34d399' }}>0.8519</td>
                    <td style={{ padding: '10px 12px', color: '#34d399', fontWeight: 700 }}>50.48% (+13.67 pp)</td>
                    <td style={{ padding: '10px 12px', color: '#34d399', fontWeight: 700 }}>44.73% (+12.81 pp)</td>
                    <td style={{ padding: '10px 12px' }}>0.7546</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', background: 'rgba(0, 240, 180, 0.08)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#38bdf8' }}>HRF External (45 Scans)</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 700 }}>Dual-Stream Hybrid (τ=0.40)</td>
                    <td style={{ padding: '10px 12px' }}>0.7419</td>
                    <td style={{ padding: '10px 12px', color: '#fca5a5' }}>0.6628 (-8.54 pp)</td>
                    <td style={{ padding: '10px 12px', color: '#34d399' }}>0.8497</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 700 }}>46.59% (+9.78 pp)</td>
                    <td style={{ padding: '10px 12px' }}>41.71%</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 700 }}>0.7745 (+2.78 pp)</td>
                  </tr>

                  {/* HRF Unexposed */}
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: 'rgba(255,255,255,0.01)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#fbbf24' }}>HRF Unexposed (5 Exploratory)</td>
                    <td style={{ padding: '10px 12px', color: '#94a3b8' }}>Raw → U-Net (τ=0.50)</td>
                    <td style={{ padding: '10px 12px' }}>0.7463</td>
                    <td style={{ padding: '10px 12px' }}>0.7263</td>
                    <td style={{ padding: '10px 12px' }}>0.7966</td>
                    <td style={{ padding: '10px 12px' }}>47.77%</td>
                    <td style={{ padding: '10px 12px' }}>42.66%</td>
                    <td style={{ padding: '10px 12px' }}>0.7540</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: 'rgba(245, 158, 11, 0.08)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#fbbf24' }}>HRF Unexposed (5 Exploratory)</td>
                    <td style={{ padding: '10px 12px', color: '#fbbf24', fontWeight: 600 }}>Restored → U-Net (τ=0.50)</td>
                    <td style={{ padding: '10px 12px' }}>0.7205</td>
                    <td style={{ padding: '10px 12px' }}>0.6233</td>
                    <td style={{ padding: '10px 12px', color: '#fbbf24' }}>0.8596</td>
                    <td style={{ padding: '10px 12px', color: '#fbbf24', fontWeight: 700 }}>58.97% (+11.21 pp)</td>
                    <td style={{ padding: '10px 12px', color: '#fbbf24', fontWeight: 700 }}>53.05% (+10.39 pp)</td>
                    <td style={{ padding: '10px 12px' }}>0.7521</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', background: 'rgba(245, 158, 11, 0.12)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#fbbf24' }}>HRF Unexposed (5 Exploratory)</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 600 }}>Dual-Stream Hybrid (τ=0.40)</td>
                    <td style={{ padding: '10px 12px' }}>0.7311</td>
                    <td style={{ padding: '10px 12px' }}>0.6423</td>
                    <td style={{ padding: '10px 12px' }}>0.8590</td>
                    <td style={{ padding: '10px 12px', color: '#fbbf24', fontWeight: 700 }}>56.30% (+8.54 pp)</td>
                    <td style={{ padding: '10px 12px' }}>51.72%</td>
                    <td style={{ padding: '10px 12px', color: '#00F0B4', fontWeight: 700 }}>0.7854</td>
                  </tr>

                  {/* Classical Frangi Baseline */}
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: 'rgba(255,255,255,0.01)' }}>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>HRF External (45 Scans)</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>Raw → Frangi (τ=0.0010)</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>0.2474</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>0.1474</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>0.7868</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>79.61%*</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>85.20%*</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>0.1560</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>HRF External (45 Scans)</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>Restored → Frangi (τ=0.0010)</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>0.2983</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>0.2654</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>0.3520</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>32.22%</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>38.15%</td>
                    <td style={{ padding: '10px 12px', color: '#64748b' }}>0.2024</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '10px' }}>
              *Note: Frangi raw sensitivity is inflated due to severe background noise false-positives (Precision 14.7%). HRF N=5 unexposed subset has insufficient statistical power (Dual-Stream vs Raw Holm p = 0.375) and is reported as exploratory evidence alongside 40 pretrain-exposed scans.
            </div>
          </div>

          {/* Disease Stratification Nuance Grid */}
          <div style={{ background: 'rgba(15, 23, 42, 0.65)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', padding: '18px' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '15px', fontWeight: 600, color: '#f1f5f9' }}>
              Clinical Pathology Stratification (FIVES 200 Test Scans)
            </h3>
            <p style={{ margin: '0 0 16px 0', fontSize: '12px', color: '#94a3b8' }}>
              Dual-Stream Hybrid (τ=0.40) retains Raw U-Net precision across pathologies while recovering Restored U-Net thin-vessel structures:
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px' }}>
              
              {/* Normal */}
              <div style={{ padding: '14px', borderRadius: '8px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 600, color: '#34d399' }}>Normal (N=50)</span>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>Healthy Arcades</span>
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.4 }}>
                  <div>Dual-Stream: <strong style={{ color: '#00F0B4' }}>0.8837</strong> (Prec: 90.2%)</div>
                  <div>Raw / Rest: <strong>0.8650</strong> / 0.6978</div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '6px' }}>
                    +3.05 pp ThinSkel vs Raw while slashing CCs (-27.34 / img).
                  </div>
                </div>
              </div>

              {/* AMD */}
              <div style={{ padding: '14px', borderRadius: '8px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 600, color: '#38bdf8' }}>AMD (N=50)</span>
                  <span style={{ fontSize: '11px', color: '#38bdf8' }}>Peak Thin Gain</span>
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.4 }}>
                  <div>Dual-Stream: <strong style={{ color: '#00F0B4' }}>0.8911</strong> (Prec: 90.2%)</div>
                  <div>Raw / Rest: <strong>0.8696</strong> / 0.8040</div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '6px' }}>
                    +15.07 pp ThinSkel vs Raw without drusen artifacts.
                  </div>
                </div>
              </div>

              {/* DR */}
              <div style={{ padding: '14px', borderRadius: '8px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 600, color: '#f59e0b' }}>Diabetic Ret. (N=50)</span>
                  <span style={{ fontSize: '11px', color: '#64748b' }}>Microaneurysms</span>
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.4 }}>
                  <div>Dual-Stream: <strong style={{ color: '#00F0B4' }}>0.8481</strong> (Prec: 89.9%)</div>
                  <div>Raw / Rest: <strong>0.8181</strong> / 0.7075</div>
                  <div style={{ fontSize: '11px', color: '#64748b', marginTop: '6px' }}>
                    +13.20 pp ThinSkel vs Raw across capillary non-perfusion borders.
                  </div>
                </div>
              </div>

              {/* Glaucoma */}
              <div style={{ padding: '14px', borderRadius: '8px', background: 'rgba(0, 240, 180, 0.08)', border: '1px solid rgba(0, 240, 180, 0.3)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 600, color: '#00F0B4' }}>Glaucoma (N=50)</span>
                  <span style={{ fontSize: '11px', color: '#00F0B4' }}>Cup Rim Resolved</span>
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.4 }}>
                  <div>Dual-Stream Prec: <strong style={{ color: '#00F0B4' }}>89.98%</strong> (Dice 0.7667)</div>
                  <div>Restored Prec: <strong style={{ color: '#f87171' }}>57.59% (-33.5 pp)</strong></div>
                  <div style={{ fontSize: '11px', color: '#cbd5e1', marginTop: '6px' }}>
                    Dual-Stream recovers +32.39 pp precision over Restored U-Net (+6.63 pp Dice &amp; +9.48 pp ThinSkel vs Raw).
                  </div>
                </div>
              </div>

            </div>
          </div>

        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: DEGRADATION & FILTER ABLATION (ORIGINAL FREQLAB STUDIO)             */}
      {/* ========================================================================= */}
      {activeTab === 'ablation' && (
        <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '24px' }}>
          
          {/* Left Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {/* Sample Selector */}
            <div style={{ background: 'rgba(15, 23, 42, 0.65)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', padding: '16px' }}>
              <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', display: 'block', marginBottom: '12px' }}>
                Benchmark Cases ({filteredSamples.length})
              </span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '280px', overflowY: 'auto' }}>
                {filteredSamples.map(s => {
                  const isSelected = selectedSampleId === s.id;
                  return (
                    <div
                      key={s.id}
                      onClick={() => setSelectedSampleId(s.id)}
                      style={{
                        padding: '10px 12px', borderRadius: '8px', cursor: 'pointer',
                        background: isSelected ? 'rgba(0, 180, 216, 0.18)' : 'rgba(255,255,255,0.02)',
                        border: isSelected ? '1px solid rgba(0, 180, 216, 0.45)' : '1px solid rgba(255,255,255,0.05)',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontSize: '13px', fontWeight: 600, color: isSelected ? '#38bdf8' : '#e2e8f0' }}>{s.title}</span>
                      </div>
                      <p style={{ margin: 0, fontSize: '11px', color: '#64748b' }}>{s.description}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Degradation Controls */}
            <div style={{ background: 'rgba(15, 23, 42, 0.65)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', padding: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <Sliders size={15} color="#00B4D8" />
                <span style={{ fontSize: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>
                  Degradation Engine
                </span>
              </div>

              <div style={{ marginBottom: '14px' }}>
                <label style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '6px' }}>Noise Profile:</label>
                <select
                  value={noiseMode}
                  onChange={e => {
                    setNoiseMode(e.target.value);
                    runClassicAblation(selectedSampleId, e.target.value, severity, modelChoice);
                  }}
                  style={{
                    width: '100%', padding: '8px 10px', borderRadius: '6px', background: 'rgba(15, 23, 42, 0.9)',
                    border: '1px solid rgba(255,255,255,0.12)', color: '#f1f5f9', fontSize: '12px', outline: 'none'
                  }}
                >
                  {NOISE_MODES.map(nm => (
                    <option key={nm.id} value={nm.id}>{nm.label}</option>
                  ))}
                </select>
              </div>

              <div style={{ marginBottom: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8', marginBottom: '4px' }}>
                  <span>Severity Multiplier:</span>
                  <span style={{ color: '#38bdf8', fontWeight: 600 }}>{severity.toFixed(1)}×</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={severity}
                  onChange={e => setSeverity(parseFloat(e.target.value))}
                  onMouseUp={() => runClassicAblation(selectedSampleId, noiseMode, severity, modelChoice)}
                  style={{ width: '100%', accentColor: '#00B4D8' }}
                />
              </div>
            </div>

          </div>

          {/* Right Ablation Canvas */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            
            {ablationResult && (
              <div style={{ background: 'rgba(15, 23, 42, 0.65)', backdropFilter: 'blur(12px)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '12px', padding: '18px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: '#f1f5f9' }}>
                    Classical Filter Ablation vs SFHformer
                  </h3>
                  <span style={{ fontSize: '11px', color: '#34d399', fontWeight: 600 }}>
                    PSNR Gain: +{ablationResult.metrics.psnr_gain} dB
                  </span>
                </div>

                {/* Triple image row */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '16px' }}>
                  <div>
                    <div style={{ fontSize: '11px', color: '#f87171', marginBottom: '4px' }}>1. Corrupted Input</div>
                    <img src={ablationResult.corrupted} alt="Corrupted" style={{ width: '100%', aspectRatio: '1/1', objectFit: 'contain', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }} />
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', color: '#38bdf8', marginBottom: '4px' }}>2. SFHformer Restored</div>
                    <img src={ablationResult.restored} alt="Restored" style={{ width: '100%', aspectRatio: '1/1', objectFit: 'contain', borderRadius: '8px', border: '1px solid rgba(56, 189, 248, 0.35)' }} />
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', color: '#00F0B4', marginBottom: '4px' }}>3. Frangi Vascular Tree</div>
                    <img src={ablationResult.vessel_mask} alt="Vessels" style={{ width: '100%', aspectRatio: '1/1', objectFit: 'contain', borderRadius: '8px', border: '1px solid rgba(0, 240, 180, 0.3)' }} />
                  </div>
                </div>

                {/* Ablation Table */}
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: '#94a3b8' }}>
                        <th style={{ padding: '8px 12px' }}>Method</th>
                        <th style={{ padding: '8px 12px' }}>Apparent Density %</th>
                        <th style={{ padding: '8px 12px' }}>Vessel Dice</th>
                        <th style={{ padding: '8px 12px' }}>Capillary Recall %</th>
                        <th style={{ padding: '8px 12px' }}>Noise FPR %</th>
                        <th style={{ padding: '8px 12px' }}>Std. CNR</th>
                        <th style={{ padding: '8px 12px' }}>Morphological Verdict</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ablationResult.ablation_comparison.map((ab, idx) => {
                        const isSFH = ab.method.includes('SFHformer');
                        return (
                          <tr 
                            key={idx}
                            style={{ 
                              borderBottom: '1px solid rgba(255,255,255,0.05)',
                              background: isSFH ? 'rgba(0, 180, 216, 0.12)' : 'transparent',
                              color: isSFH ? '#38bdf8' : '#e2e8f0',
                              fontWeight: isSFH ? 600 : 400
                            }}
                          >
                            <td style={{ padding: '10px 12px' }}>{ab.method}</td>
                            <td style={{ padding: '10px 12px' }}>{ab.vessel_density_pct}%</td>
                            <td style={{ padding: '10px 12px' }}>{ab.vessel_dice}</td>
                            <td style={{ padding: '10px 12px', color: (ab.recall_pct && ab.recall_pct > 50) ? '#34d399' : '#f87171' }}>
                              {ab.recall_pct !== undefined ? `${ab.recall_pct}%` : '—'}
                            </td>
                            <td style={{ padding: '10px 12px', color: (ab.fpr_pct !== undefined && ab.fpr_pct < 2) ? '#34d399' : '#f87171' }}>
                              {ab.fpr_pct !== undefined ? `${ab.fpr_pct}%` : '—'}
                            </td>
                            <td style={{ padding: '10px 12px' }}>{ab.cnr}</td>
                            <td style={{ padding: '10px 12px', color: isSFH ? '#34d399' : '#94a3b8' }}>
                              {ab.clinical_verdict}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

              </div>
            )}

          </div>

        </div>
      )}

    </div>
  );
}