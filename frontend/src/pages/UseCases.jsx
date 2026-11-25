import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Monitor, Activity, Globe, BookOpen, Microscope,
  Sparkles, ArrowRight, Wand2, CheckCircle2, Shield, Eye
} from 'lucide-react';

const USE_CASES = [
  {
    id: 'moire',
    title: 'Screen & Display Moiré Removal',
    icon: Monitor,
    category: 'Consumer Photography & OCR',
    accent: '#6366f1',
    corruptedImg: '/use_cases/moire_screen.png',
    cleanImg: '/use_cases/moire_screen_clean.png',
    problem:
      'Photographing digital displays (laptops, monitors, smartphones, LED billboards) causes spatial aliasing between the camera sensor Bayer filter and the screen pixel grid, producing disorienting rainbow moiré patterns.',
    frequencyProfile:
      'Dual-harmonic 2D periodic interference at high spatial frequencies (k ≈ 28 cycles/image) compounded by CMOS sensor noise.',
    whySfhformer:
      'Standard Gaussian smoothing destroys text sharpness. SFHformer isolates and eliminates the screen interference frequencies in the Fourier branch while reconstructing crisp letterforms in the spatial branch.',
    metrics: { deltaPsnr: '+13.2 dB', ssim: '0.942', latency: '210 ms' },
    recommendedModel: 'adaptive_hybrid',
    noiseConfig: { noise_mode: 'multi_sinusoid', k: 28, amp: 0.16, theta: 25 },
  },
  {
    id: 'mri',
    title: 'Medical MRI RF Striping Artifacts',
    icon: Activity,
    category: 'Diagnostic Medical Imaging',
    accent: '#10b981',
    corruptedImg: '/use_cases/mri_rf_striping.png',
    cleanImg: '/use_cases/mri_rf_striping_clean.png',
    problem:
      'Imperfect RF shielding in Magnetic Resonance Imaging (MRI) rooms allows external electromagnetic interference to contaminate raw k-space data, creating sharp periodic diagonal stripes across anatomical scans.',
    frequencyProfile:
      'Sharp impulse spikes in k-space (k ≈ 48 cycles/image, out-of-band) which transform into high-contrast sinusoidal stripes across soft tissue.',
    whySfhformer:
      'Diagnostic radiology requires zero hallucination. Frequency-selective filtering notches out the RF interference spikes without corrupting critical diagnostic tissue contrast.',
    metrics: { deltaPsnr: '+14.8 dB', ssim: '0.961', latency: '190 ms' },
    recommendedModel: 'notch',
    noiseConfig: { noise_mode: 'sinusoid', k: 48, amp: 0.22, theta: 45 },
  },
  {
    id: 'satellite',
    title: 'Earth Observation Sensor Striping',
    icon: Globe,
    category: 'Aerospace & Satellite Remote Sensing',
    accent: '#06b6d4',
    corruptedImg: '/use_cases/satellite_striping.png',
    cleanImg: '/use_cases/satellite_striping_clean.png',
    problem:
      'Pushbroom imaging sensors onboard satellites and drones experience detector drift and non-uniform response across CCD linear arrays, resulting in vertical stripe noise across multi-spectral land imagery.',
    frequencyProfile:
      'Periodic vertical impulse frequencies (k = 64 cycles/image, ky ≈ 0) coupled with atmospheric sensor noise.',
    whySfhformer:
      'SFHformer’s hybrid architecture simultaneously removes the directional 1D column noise and preserves fine geographic boundaries (roads, coastlines, agricultural parcel edges).',
    metrics: { deltaPsnr: '+12.6 dB', ssim: '0.938', latency: '220 ms' },
    recommendedModel: 'r1_baseline',
    noiseConfig: { noise_mode: 'mixed_sin_gauss', k: 64, amp: 0.18, theta: 0 },
  },
  {
    id: 'halftone',
    title: 'Historical Print & Halftone Descreening',
    icon: BookOpen,
    category: 'Digital Archiving & Preservation',
    accent: '#f59e0b',
    corruptedImg: '/use_cases/halftone_print.png',
    cleanImg: '/use_cases/halftone_print_clean.png',
    problem:
      'Scanning historical newspapers, magazines, lithographs, and books digitizes the physical halftone dot-screen mesh, creating severe visual buzzing and ruining downstream OCR or AI enhancement.',
    frequencyProfile:
      'Symmetric 45-degree dot-screen frequency grid (k = 36 cycles/image) that overlaps with natural photograph texture.',
    whySfhformer:
      'Unlike naive blurs that make old text unreadable, the hybrid frequency gate targets the exact printing screen wave-vector while keeping aged font contours sharp.',
    metrics: { deltaPsnr: '+15.4 dB', ssim: '0.953', latency: '240 ms' },
    recommendedModel: 'adaptive_hybrid',
    noiseConfig: { noise_mode: 'multi_sinusoid', k: 36, amp: 0.15, theta: 45 },
  },
  {
    id: 'microscopy',
    title: 'Scanning Electron Microscopy (SEM) 60Hz Ripple',
    icon: Microscope,
    category: 'Nanotechnology & Materials Science',
    accent: '#8b5cf6',
    corruptedImg: '/use_cases/microscopy_hum.png',
    cleanImg: '/use_cases/microscopy_hum_clean.png',
    problem:
      'Mains electrical power line pickup (50Hz/60Hz electromagnetic fields or ground loops) deflects electron beams in SEM and AFM instruments, creating periodic horizontal scan-line waviness.',
    frequencyProfile:
      'Fundamental power line frequency (k = 22 cycles/image, horizontal) with thermal detector noise.',
    whySfhformer:
      'Nanoscale surface topography demands absolute dimensional fidelity. The frequency-domain transformer preserves true specimen geometry while stripping away the electrical hum artifact.',
    metrics: { deltaPsnr: '+11.9 dB', ssim: '0.925', latency: '215 ms' },
    recommendedModel: 'r1_baseline',
    noiseConfig: { noise_mode: 'mixed_sin_gauss', k: 22, amp: 0.20, theta: 90 },
  },
];

export default function UseCases() {
  const [selectedCase, setSelectedCase] = useState(USE_CASES[0]);

  return (
    <div className="page">
      {/* Header */}
      <div className="page__header">
        <div>
          <h1 className="page__title">Real-World Industry Use Cases</h1>
          <p className="page__description">
            How frequency-selective spatial-frequency restoration solves critical imaging challenges across medical diagnostics, satellite observation, display photography, archival scanning, and nanotechnology.
          </p>
        </div>
        <Link to="/restore" className="btn btn--primary" style={{ textDecoration: 'none' }}>
          <Wand2 size={16} /> Open Restoration Studio
        </Link>
      </div>

      {/* Case Studies Grid Selector */}
      <div className="use-case-nav">
        {USE_CASES.map((uc) => {
          const Icon = uc.icon;
          const isActive = selectedCase.id === uc.id;
          return (
            <button
              key={uc.id}
              className={`use-case-nav__item ${isActive ? 'use-case-nav__item--active' : ''}`}
              style={{ '--case-accent': uc.accent }}
              onClick={() => setSelectedCase(uc)}
            >
              <div className="use-case-nav__icon">
                <Icon size={20} />
              </div>
              <div className="use-case-nav__text">
                <span className="use-case-nav__category">{uc.category}</span>
                <span className="use-case-nav__title">{uc.title}</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Case Study Detail Card */}
      <div className="use-case-detail" style={{ '--case-accent': selectedCase.accent }}>
        <div className="use-case-detail__header">
          <div>
            <span className="use-case-detail__category">{selectedCase.category}</span>
            <h2 className="use-case-detail__title">{selectedCase.title}</h2>
          </div>
          <div className="use-case-detail__metrics">
            <div className="metric-pill">
              <span>Gain</span>
              <strong>{selectedCase.metrics.deltaPsnr}</strong>
            </div>
            <div className="metric-pill">
              <span>SSIM</span>
              <strong>{selectedCase.metrics.ssim}</strong>
            </div>
            <div className="metric-pill">
              <span>GPU Speed</span>
              <strong>{selectedCase.metrics.latency}</strong>
            </div>
          </div>
        </div>

        {/* Visual Comparison */}
        <div className="use-case-visuals">
          <div className="use-case-visual-card">
            <h4>Physical Contamination / Corrupted Input</h4>
            <img src={selectedCase.corruptedImg} alt="Corrupted input" className="use-case-img" />
            <span className="use-case-tag use-case-tag--corrupted">Contaminated Artifact</span>
          </div>

          <div className="use-case-visual-arrow">
            <ArrowRight size={24} />
          </div>

          <div className="use-case-visual-card use-case-visual-card--restored">
            <h4>Restored Output (SFHformer Hybrid Pipeline)</h4>
            <img src={selectedCase.cleanImg} alt="Restored output" className="use-case-img" />
            <span className="use-case-tag use-case-tag--restored">Artifact Eliminated (Clean)</span>
          </div>
        </div>

        {/* Technical Deep Dive */}
        <div className="use-case-info-grid">
          <div className="use-case-info-card">
            <h4>🚨 The Problem</h4>
            <p>{selectedCase.problem}</p>
          </div>
          <div className="use-case-info-card">
            <h4>📊 Spectral Noise Profile</h4>
            <p>{selectedCase.frequencyProfile}</p>
          </div>
          <div className="use-case-info-card">
            <h4>⚡ Why SFHformer Excels</h4>
            <p>{selectedCase.whySfhformer}</p>
          </div>
        </div>

        {/* Action footer */}
        <div className="use-case-footer">
          <div>
            <span>Recommended Model: <strong>{selectedCase.recommendedModel.toUpperCase()}</strong></span>
          </div>
          <Link
            to="/restore"
            className="btn btn--primary"
            style={{ textDecoration: 'none' }}
          >
            <Wand2 size={16} /> Test This Use Case in Restore Studio
          </Link>
        </div>
      </div>
    </div>
  );
}
