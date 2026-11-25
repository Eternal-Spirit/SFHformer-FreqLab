import { useState } from 'react';
import { HEATMAP_IMAGES, SPECTRUM_IMAGES, FREQ_RESPONSE_IMAGE, METHODS } from '../data/experimentData';
import { Image, Maximize2, X } from 'lucide-react';

function ImageViewer({ src, alt, onClose }) {
  return (
    <div className="lightbox" onClick={onClose}>
      <div className="lightbox__content" onClick={(e) => e.stopPropagation()}>
        <button className="lightbox__close" onClick={onClose}><X size={20} /></button>
        <img src={src} alt={alt} className="lightbox__image" />
        <p className="lightbox__caption">{alt}</p>
      </div>
    </div>
  );
}

export default function Visualizations() {
  const [heatmapType, setHeatmapType] = useState('freq_orient');
  const [selectedMethod, setSelectedMethod] = useState('r1_baseline');
  const [lightbox, setLightbox] = useState(null);
  const [compareMode, setCompareMode] = useState(false);
  const [compareMethods, setCompareMethods] = useState(['r1_baseline', 'notch']);

  const heatmapLabels = {
    freq_orient: 'Frequency × Orientation',
    freq_amp: 'Frequency × Amplitude',
  };

  const toggleCompare = (mid) => {
    setCompareMethods((prev) => {
      if (prev.includes(mid)) return prev.filter((x) => x !== mid);
      if (prev.length >= 4) return prev;
      return [...prev, mid];
    });
  };

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1 className="page__title">Visualizations</h1>
          <p className="page__description">
            Heatmaps, frequency response curves, and spectral analysis figures
          </p>
        </div>
        <button
          className={`btn ${compareMode ? 'btn--primary' : 'btn--secondary'}`}
          onClick={() => setCompareMode(!compareMode)}
        >
          {compareMode ? '✕ Exit Compare' : '⚡ Compare Mode'}
        </button>
      </div>

      {/* Frequency Response Overview */}
      <div className="viz-section">
        <h2 className="viz-section__title">
          <Image size={18} /> Frequency Response Overview
        </h2>
        <div className="viz-section__image-wrap">
          <img
            src={`/images/${FREQ_RESPONSE_IMAGE}`}
            alt="Frequency Response"
            className="viz-section__image"
            onClick={() => setLightbox({ src: `/images/${FREQ_RESPONSE_IMAGE}`, alt: 'Frequency Response' })}
          />
          <button
            className="viz-section__expand"
            onClick={() => setLightbox({ src: `/images/${FREQ_RESPONSE_IMAGE}`, alt: 'Frequency Response' })}
          >
            <Maximize2 size={16} />
          </button>
        </div>
      </div>

      {/* Heatmaps */}
      <div className="viz-section">
        <h2 className="viz-section__title">ΔPSNR Heatmaps</h2>
        <div className="viz-controls">
          <div className="filter-bar__section">
            <span className="filter-bar__label">Heatmap Type</span>
            <div className="filter-bar__chips">
              {Object.entries(heatmapLabels).map(([key, label]) => (
                <button
                  key={key}
                  className={`chip ${heatmapType === key ? 'chip--active' : ''}`}
                  onClick={() => setHeatmapType(key)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {!compareMode ? (
          <>
            <div className="filter-bar__section" style={{ marginBottom: '1rem' }}>
              <span className="filter-bar__label">Method</span>
              <div className="filter-bar__chips">
                {METHODS.filter((m) => m.id !== 'gaussian_s2.0').map((m) => (
                  <button
                    key={m.id}
                    className={`chip ${selectedMethod === m.id ? 'chip--active' : ''}`}
                    style={selectedMethod === m.id ? { '--chip-color': m.color } : {}}
                    onClick={() => setSelectedMethod(m.id)}
                  >
                    <span className="chip__dot" style={{ background: m.color }}></span>
                    {m.name}
                  </button>
                ))}
              </div>
            </div>
            <div className="viz-section__image-wrap viz-section__image-wrap--large">
              <img
                src={`/images/${HEATMAP_IMAGES[heatmapType][selectedMethod]}`}
                alt={`${selectedMethod} - ${heatmapLabels[heatmapType]}`}
                className="viz-section__image"
                onClick={() =>
                  setLightbox({
                    src: `/images/${HEATMAP_IMAGES[heatmapType][selectedMethod]}`,
                    alt: `${selectedMethod} - ${heatmapLabels[heatmapType]}`,
                  })
                }
              />
            </div>
          </>
        ) : (
          <>
            <div className="filter-bar__section" style={{ marginBottom: '1rem' }}>
              <span className="filter-bar__label">Select up to 4 methods</span>
              <div className="filter-bar__chips">
                {METHODS.filter((m) => m.id !== 'gaussian_s2.0').map((m) => (
                  <button
                    key={m.id}
                    className={`chip ${compareMethods.includes(m.id) ? 'chip--active' : ''}`}
                    style={compareMethods.includes(m.id) ? { '--chip-color': m.color } : {}}
                    onClick={() => toggleCompare(m.id)}
                  >
                    <span className="chip__dot" style={{ background: m.color }}></span>
                    {m.name}
                  </button>
                ))}
              </div>
            </div>
            <div className="compare-grid">
              {compareMethods.map((mid) => {
                const m = METHODS.find((mm) => mm.id === mid);
                return (
                  <div key={mid} className="compare-grid__item">
                    <h4 className="compare-grid__label">{m?.name}</h4>
                    <img
                      src={`/images/${HEATMAP_IMAGES[heatmapType][mid]}`}
                      alt={`${mid} - ${heatmapLabels[heatmapType]}`}
                      className="compare-grid__image"
                      onClick={() =>
                        setLightbox({
                          src: `/images/${HEATMAP_IMAGES[heatmapType][mid]}`,
                          alt: `${mid} - ${heatmapLabels[heatmapType]}`,
                        })
                      }
                    />
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>

      {/* Spectrum Figures */}
      <div className="viz-section">
        <h2 className="viz-section__title">Log-Magnitude Spectrum Comparisons</h2>
        <div className="spectrum-grid">
          {SPECTRUM_IMAGES.map((img) => (
            <div key={img.id} className="spectrum-grid__item">
              <h4 className="spectrum-grid__label">{img.label}</h4>
              <img
                src={`/images/${img.file}`}
                alt={img.label}
                className="spectrum-grid__image"
                onClick={() => setLightbox({ src: `/images/${img.file}`, alt: img.label })}
              />
            </div>
          ))}
        </div>
      </div>

      {lightbox && <ImageViewer src={lightbox.src} alt={lightbox.alt} onClose={() => setLightbox(null)} />}
    </div>
  );
}
