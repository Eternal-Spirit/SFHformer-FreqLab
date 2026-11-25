import { useState } from 'react';
import {
  TRAIN_LOG_R1, TRAIN_LOG_R2, VAL_LOG_R1, VAL_LOG_R2,
} from '../data/experimentData';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Area, AreaChart, ComposedChart, Bar,
} from 'recharts';
import { useSettings } from '../context/SettingsContext';
import { Cpu, Clock, HardDrive, Gauge } from 'lucide-react';

export default function TrainingMonitor() {
  const { settings } = useSettings();
  const [selectedModel, setSelectedModel] = useState('both');
  const [chartType, setChartType] = useState('loss');

  // Combined loss data
  const lossData = TRAIN_LOG_R1.map((r1) => {
    const r2 = TRAIN_LOG_R2.find((r) => r.iter === r1.iter);
    return {
      iter: r1.iter,
      'R1 Baseline': r1.loss,
      ...(r2 ? { 'R2 Gated': r2.loss } : {}),
    };
  });

  // Combined val PSNR data
  const valData = VAL_LOG_R1.map((r1) => {
    const r2 = VAL_LOG_R2.find((r) => r.iter === r1.iter);
    return {
      iter: r1.iter,
      'R1 Baseline': r1.val_psnr,
      ...(r2 ? { 'R2 Gated': r2.val_psnr } : {}),
    };
  });

  // Learning rate data
  const lrData = TRAIN_LOG_R1.map((r1) => {
    const r2 = TRAIN_LOG_R2.find((r) => r.iter === r1.iter);
    return {
      iter: r1.iter,
      'R1 Baseline': r1.lr,
      ...(r2 ? { 'R2 Gated': r2.lr } : {}),
    };
  });

  // Val loss data
  const valLossData = VAL_LOG_R1.map((r1) => {
    const r2 = VAL_LOG_R2.find((r) => r.iter === r1.iter);
    return {
      iter: r1.iter,
      'R1 Baseline': r1.val_loss,
      ...(r2 ? { 'R2 Gated': r2.val_loss } : {}),
    };
  });

  const showR1 = selectedModel === 'both' || selectedModel === 'r1';
  const showR2 = selectedModel === 'both' || selectedModel === 'r2';

  const r1Final = TRAIN_LOG_R1[TRAIN_LOG_R1.length - 1];
  const r2Final = TRAIN_LOG_R2[TRAIN_LOG_R2.length - 1];
  const r1ValFinal = VAL_LOG_R1[VAL_LOG_R1.length - 1];
  const r2ValFinal = VAL_LOG_R2[VAL_LOG_R2.length - 1];

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1 className="page__title">Training Monitor</h1>
          <p className="page__description">
            Training loss, validation PSNR, learning rate schedules, and hardware utilization
          </p>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="stats-grid">
        <div className="stat-card" style={{ '--card-accent': '#6366f1' }}>
          <div className="stat-card__icon"><Gauge size={22} /></div>
          <div className="stat-card__content">
            <span className="stat-card__label">R1 Final PSNR</span>
            <span className="stat-card__value">{r1ValFinal.val_psnr}<span className="stat-card__unit">dB</span></span>
            <span className="stat-card__subtitle">Loss: {r1Final.loss.toFixed(4)}</span>
          </div>
        </div>
        <div className="stat-card" style={{ '--card-accent': '#8b5cf6' }}>
          <div className="stat-card__icon"><Gauge size={22} /></div>
          <div className="stat-card__content">
            <span className="stat-card__label">R2 Final PSNR</span>
            <span className="stat-card__value">{r2ValFinal.val_psnr}<span className="stat-card__unit">dB</span></span>
            <span className="stat-card__subtitle">Loss: {r2Final.loss.toFixed(4)}</span>
          </div>
        </div>
        <div className="stat-card" style={{ '--card-accent': '#10b981' }}>
          <div className="stat-card__icon"><Clock size={22} /></div>
          <div className="stat-card__content">
            <span className="stat-card__label">R1 Train Time</span>
            <span className="stat-card__value">{(r1Final.elapsed / 3600).toFixed(1)}<span className="stat-card__unit">hrs</span></span>
            <span className="stat-card__subtitle">{(r1Final.iter / r1Final.elapsed).toFixed(1)} it/s</span>
          </div>
        </div>
        <div className="stat-card" style={{ '--card-accent': '#f59e0b' }}>
          <div className="stat-card__icon"><HardDrive size={22} /></div>
          <div className="stat-card__content">
            <span className="stat-card__label">Peak VRAM</span>
            <span className="stat-card__value">{r2Final.vram}<span className="stat-card__unit">GB</span></span>
            <span className="stat-card__subtitle">RTX 5060 (8 GB)</span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="filter-bar">
        <div className="filter-bar__section">
          <span className="filter-bar__label">Model</span>
          <div className="filter-bar__chips">
            {[
              { id: 'both', label: 'Both Models' },
              { id: 'r1', label: 'R1 Baseline' },
              { id: 'r2', label: 'R2 Gated' },
            ].map((opt) => (
              <button
                key={opt.id}
                className={`chip ${selectedModel === opt.id ? 'chip--active' : ''}`}
                onClick={() => setSelectedModel(opt.id)}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <div className="filter-bar__section">
          <span className="filter-bar__label">Chart</span>
          <div className="filter-bar__chips">
            {[
              { id: 'loss', label: 'Training Loss' },
              { id: 'val_psnr', label: 'Val PSNR' },
              { id: 'val_loss', label: 'Val Loss' },
              { id: 'lr', label: 'Learning Rate' },
            ].map((opt) => (
              <button
                key={opt.id}
                className={`chip ${chartType === opt.id ? 'chip--active' : ''}`}
                onClick={() => setChartType(opt.id)}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Charts */}
      {chartType === 'loss' && (
        <div className="chart-container">
          <h3 className="chart-container__title">Training Loss</h3>
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={lossData}>
              {settings.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />}
              <XAxis
                dataKey="iter"
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
                label={{ value: 'Iteration', position: 'bottom', fill: 'var(--text-secondary)' }}
              />
              <YAxis
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
                label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                }}
              />
              <Legend />
              {showR1 && (
                <Area
                  type="monotone" dataKey="R1 Baseline" stroke="#6366f1"
                  fill="#6366f1" fillOpacity={0.1} strokeWidth={2} dot={false}
                />
              )}
              {showR2 && (
                <Area
                  type="monotone" dataKey="R2 Gated" stroke="#8b5cf6"
                  fill="#8b5cf6" fillOpacity={0.1} strokeWidth={2} dot={false}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {chartType === 'val_psnr' && (
        <div className="chart-container">
          <h3 className="chart-container__title">Validation PSNR</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={valData}>
              {settings.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />}
              <XAxis
                dataKey="iter"
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
              />
              <YAxis
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
                domain={['auto', 'auto']}
                label={{ value: 'PSNR (dB)', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                }}
              />
              <Legend />
              {showR1 && (
                <Line
                  type="monotone" dataKey="R1 Baseline" stroke="#6366f1"
                  strokeWidth={2.5} dot={{ r: 4, fill: '#6366f1' }}
                  activeDot={{ r: 6 }}
                />
              )}
              {showR2 && (
                <Line
                  type="monotone" dataKey="R2 Gated" stroke="#8b5cf6"
                  strokeWidth={2.5} dot={{ r: 4, fill: '#8b5cf6' }}
                  activeDot={{ r: 6 }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {chartType === 'val_loss' && (
        <div className="chart-container">
          <h3 className="chart-container__title">Validation Loss</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={valLossData}>
              {settings.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />}
              <XAxis
                dataKey="iter"
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
              />
              <YAxis
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
                label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                }}
              />
              <Legend />
              {showR1 && <Line type="monotone" dataKey="R1 Baseline" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 4, fill: '#6366f1' }} />}
              {showR2 && <Line type="monotone" dataKey="R2 Gated" stroke="#8b5cf6" strokeWidth={2.5} dot={{ r: 4, fill: '#8b5cf6' }} />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {chartType === 'lr' && (
        <div className="chart-container">
          <h3 className="chart-container__title">Learning Rate Schedule (Cosine Annealing)</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={lrData}>
              {settings.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />}
              <XAxis
                dataKey="iter"
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
              />
              <YAxis
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickFormatter={(v) => v.toExponential(1)}
                label={{ value: 'LR', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                }}
                formatter={(v) => v.toExponential(2)}
              />
              <Legend />
              {showR1 && <Line type="monotone" dataKey="R1 Baseline" stroke="#6366f1" strokeWidth={2} dot={false} />}
              {showR2 && <Line type="monotone" dataKey="R2 Gated" stroke="#8b5cf6" strokeWidth={2} dot={false} />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
