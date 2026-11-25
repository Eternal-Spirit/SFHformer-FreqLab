import {
  PROJECT_META,
  HYPOTHESES,
  SUMMARY_BAND,
  DO_NO_HARM,
  METHODS,
  getMethodColor,
} from '../data/experimentData';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, ErrorBar,
} from 'recharts';
import { useSettings } from '../context/SettingsContext';
import { Link } from 'react-router-dom';
import { Activity, Zap, Target, Shield, TrendingUp, TrendingDown, Minus, Wand2, Briefcase } from 'lucide-react';

function StatCard({ icon: Icon, label, value, unit, accent, subtitle }) {
  return (
    <div className="stat-card" style={{ '--card-accent': accent }}>
      <div className="stat-card__icon">
        <Icon size={22} />
      </div>
      <div className="stat-card__content">
        <span className="stat-card__label">{label}</span>
        <span className="stat-card__value">
          {value}
          {unit && <span className="stat-card__unit">{unit}</span>}
        </span>
        {subtitle && <span className="stat-card__subtitle">{subtitle}</span>}
      </div>
    </div>
  );
}

function HypothesisCard({ h }) {
  const isConfirmed = h.verdict === 'CONFIRMED';
  return (
    <div className={`hypothesis-card hypothesis-card--${isConfirmed ? 'confirmed' : 'rejected'}`}>
      <div className="hypothesis-card__header">
        <span className="hypothesis-card__id">{h.id}</span>
        <span className={`hypothesis-card__badge hypothesis-card__badge--${isConfirmed ? 'confirmed' : 'rejected'}`}>
          {h.icon} {h.verdict}
        </span>
      </div>
      <h3 className="hypothesis-card__title">{h.title}</h3>
      <div className="hypothesis-card__row">
        <span className="hypothesis-card__label">Prediction</span>
        <p className="hypothesis-card__text">{h.prediction}</p>
      </div>
      <div className="hypothesis-card__row">
        <span className="hypothesis-card__label">Result</span>
        <p className="hypothesis-card__text hypothesis-card__text--result">{h.result}</p>
      </div>
      <p className="hypothesis-card__detail">{h.details}</p>
    </div>
  );
}

function BandComparisonChart() {
  const { settings } = useSettings();
  const methods = ['r1_baseline', 'r2_gated', 'notch', 'gaussian_s1.0', 'median_3'];

  const data = methods.map((mid) => {
    const inBand = SUMMARY_BAND.find((s) => s.method === mid && s.band === 'in');
    const outBand = SUMMARY_BAND.find((s) => s.method === mid && s.band === 'out');
    const m = METHODS.find((m) => m.id === mid);
    return {
      name: m?.name || mid,
      'In-Band': inBand?.mean || 0,
      'Out-of-Band': outBand?.mean || 0,
      inErr: inBand ? [inBand.mean - inBand.ci_lo, inBand.ci_hi - inBand.mean] : [0, 0],
      outErr: outBand ? [outBand.mean - outBand.ci_lo, outBand.ci_hi - outBand.mean] : [0, 0],
    };
  });

  return (
    <div className="chart-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h3 className="chart-container__title" style={{ margin: 0 }}>
          In-Band vs. Out-of-Band Restoration Gain: ΔPSNR (dB)
        </h3>
        <span style={{ fontSize: '11px', color: 'var(--text-secondary)', background: 'var(--surface-elevated)', padding: '2px 8px', borderRadius: '4px', border: '1px solid var(--border)' }}>
          ΔPSNR = PSNR<sub>restored</sub> − PSNR<sub>corrupted</sub>
        </span>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} barGap={4} barCategoryGap="20%">
          {settings.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />}
          <XAxis
            dataKey="name"
            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
            axisLine={{ stroke: 'var(--border)' }}
          />
          <YAxis
            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
            axisLine={{ stroke: 'var(--border)' }}
            label={{ value: 'ΔPSNR (dB)', angle: -90, position: 'insideLeft', fill: 'var(--text-secondary)' }}
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
          <Bar dataKey="In-Band" fill="#6366f1" radius={[4, 4, 0, 0]}>
            {settings.showConfidenceIntervals && (
              <ErrorBar dataKey="inErr" width={4} strokeWidth={1.5} stroke="#6366f1" />
            )}
          </Bar>
          <Bar dataKey="Out-of-Band" fill="#f59e0b" radius={[4, 4, 0, 0]}>
            {settings.showConfidenceIntervals && (
              <ErrorBar dataKey="outErr" width={4} strokeWidth={1.5} stroke="#f59e0b" />
            )}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ marginTop: '10px', padding: '10px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
        <strong>Statistical Protocol:</strong> N = 50 test crops (DIV2K Valid HR, zero leakage). Error bars denote <strong>95% Bootstrap Confidence Intervals</strong> (10,000 resamples). In-Band: <em>k</em> ∈ [8, 38] cycles/image; Out-of-Band: <em>k</em> ∈ [1, 7] ∪ [39, 120]. Paired bootstrap test between R1 and R2 out-of-band yields <em>p</em> = 0.28 (statistically indistinguishable, corroborating H2 rejection).
      </div>
    </div>
  );
}

function DoNoHarmTable() {
  const { settings } = useSettings();
  const dp = settings.decimalPlaces;
  const sorted = [...DO_NO_HARM].sort((a, b) => b.psnr - a.psnr);

  return (
    <div className="chart-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h3 className="chart-container__title" style={{ margin: 0 }}>
          <Shield size={18} /> Do-No-Harm Test (Clean Inputs)
        </h3>
        <span style={{ fontSize: '11px', color: '#10b981', background: 'rgba(16, 185, 129, 0.1)', padding: '2px 8px', borderRadius: '4px', border: '1px solid rgba(16, 185, 129, 0.25)' }}>
          Predefined: Safe if PSNR ≥ 40 dB & SSIM ≥ 0.9700
        </span>
      </div>
      <table className={`data-table ${settings.compactTables ? 'data-table--compact' : ''}`}>
        <thead>
          <tr>
            <th>Method</th>
            <th>Type</th>
            <th>PSNR (dB)</th>
            <th>SSIM</th>
            <th>Status Criteria</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const m = METHODS.find((m) => m.id === row.method);
            const isSafe = row.psnr >= 40.0 && row.ssim >= 0.9700;
            const isWarn = !isSafe && (row.psnr >= 30.0 || row.ssim >= 0.8500);
            return (
              <tr key={row.method}>
                <td>
                  <span className="method-dot" style={{ background: m?.color }}></span>
                  {m?.name || row.method}
                </td>
                <td>
                  <span className={`type-badge type-badge--${m?.type}`}>{m?.type}</span>
                </td>
                <td className="mono">{row.psnr.toFixed(dp)}</td>
                <td className="mono">{row.ssim.toFixed(4)}</td>
                <td>
                  {isSafe ? (
                    <span className="status-badge status-badge--safe">✓ Safe (Pass)</span>
                  ) : isWarn ? (
                    <span className="status-badge status-badge--warn">⚠ Moderate</span>
                  ) : (
                    <span className="status-badge status-badge--danger">✗ Harmful (Ablated)</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{ marginTop: '10px', padding: '10px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid var(--border)', fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
        <strong>Metric Notes:</strong> Identity MSE = 0 mathematically yields infinite PSNR; capped at 100.00 dB for numerical formatting. Notch filter achieves 73.77 dB PSNR (MSE &lt; 10<sup>−7</sup>) with SSIM 0.9733 due to minute frequency notch boundary ringing affecting local structural correlation windows. Clean-image preservation is a baseline prerequisite; clinical safety also requires downstream vessel biomarker validation.
      </div>
    </div>
  );
}

export default function Dashboard() {
  const r1In = SUMMARY_BAND.find((s) => s.method === 'r1_baseline' && s.band === 'in');
  const r1Out = SUMMARY_BAND.find((s) => s.method === 'r1_baseline' && s.band === 'out');
  const gap = (r1In.mean - r1Out.mean).toFixed(2);

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1 className="page__title">Dashboard</h1>
          <p className="page__description">{PROJECT_META.question}</p>
        </div>
        <div className="page__header-badges" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Link to="/use-cases" className="btn btn--secondary" style={{ textDecoration: 'none' }}>
            <Briefcase size={16} /> Industry Use Cases
          </Link>
          <Link to="/restore" className="btn btn--primary" style={{ textDecoration: 'none' }}>
            <Wand2 size={16} /> Open Restore Studio
          </Link>
          <span className="badge badge--primary">{PROJECT_META.testConditions.toLocaleString()} Evaluations</span>
          <span className="badge badge--secondary">{PROJECT_META.testCrops} Test Crops</span>
        </div>
      </div>

      <div className="stats-grid">
        <StatCard
          icon={TrendingUp}
          label="Best In-Band ΔPSNR"
          value={r1In.mean.toFixed(2)}
          unit="dB"
          accent="#6366f1"
          subtitle="R1 Baseline (k ∈ [8, 38])"
        />
        <StatCard
          icon={TrendingDown}
          label="Out-of-Band Drop"
          value={`−${gap}`}
          unit="dB"
          accent="#ef4444"
          subtitle="Logarithmic drop (not linear %)"
        />
        <StatCard
          icon={Zap}
          label="Notch Filter OOB"
          value="13.41"
          unit="dB"
          accent="#10b981"
          subtitle="Beats neural by +6.33 dB on sine"
        />
        <StatCard
          icon={Activity}
          label="Training Iterations"
          value="20K"
          unit="per model"
          accent="#f59e0b"
          subtitle="~2.6 hrs total"
        />
      </div>

      <div className="section-title">
        <Target size={20} />
        <h2>Pre-Registered Hypotheses</h2>
        <div className="section-title__line"></div>
      </div>

      <div className="hypotheses-grid">
        {HYPOTHESES.map((h) => (
          <HypothesisCard key={h.id} h={h} />
        ))}
      </div>

      <div className="dashboard-charts">
        <BandComparisonChart />
        <DoNoHarmTable />
      </div>

      <div className="dashboard-footer">
        <div className="dashboard-footer__meta">
          <span>📅 {PROJECT_META.dateCompleted}</span>
          <span>💻 {PROJECT_META.hardware}</span>
          <span>📊 {PROJECT_META.dataset}</span>
          <span>🎯 Band: {PROJECT_META.trainingBand}</span>
        </div>
      </div>
    </div>
  );
}
