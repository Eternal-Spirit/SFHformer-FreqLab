import { useSettings } from '../context/SettingsContext';
import { Settings as SettingsIcon, RotateCcw, Palette, Type, Eye, BarChart3 } from 'lucide-react';

const ACCENT_COLORS = [
  { label: 'Indigo', value: '#6366f1' },
  { label: 'Violet', value: '#8b5cf6' },
  { label: 'Emerald', value: '#10b981' },
  { label: 'Amber', value: '#f59e0b' },
  { label: 'Rose', value: '#f43f5e' },
  { label: 'Cyan', value: '#06b6d4' },
  { label: 'Blue', value: '#3b82f6' },
];

function SettingRow({ label, description, children }) {
  return (
    <div className="setting-row">
      <div className="setting-row__info">
        <span className="setting-row__label">{label}</span>
        {description && <span className="setting-row__description">{description}</span>}
      </div>
      <div className="setting-row__control">{children}</div>
    </div>
  );
}

function Toggle({ checked, onChange }) {
  return (
    <button
      className={`toggle ${checked ? 'toggle--on' : ''}`}
      onClick={() => onChange(!checked)}
      role="switch"
      aria-checked={checked}
    >
      <span className="toggle__thumb"></span>
    </button>
  );
}

export default function SettingsPage() {
  const { settings, updateSetting, resetSettings } = useSettings();

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1 className="page__title">Settings</h1>
          <p className="page__description">Customize the dashboard appearance and behavior</p>
        </div>
        <button className="btn btn--danger" onClick={resetSettings}>
          <RotateCcw size={16} /> Reset All
        </button>
      </div>

      <div className="settings-grid">
        {/* Appearance */}
        <div className="settings-group">
          <h3 className="settings-group__title"><Palette size={18} /> Appearance</h3>

          <SettingRow label="Theme" description="Switch between dark and light mode">
            <div className="filter-bar__chips">
              {['dark', 'light'].map((t) => (
                <button
                  key={t}
                  className={`chip ${settings.theme === t ? 'chip--active' : ''}`}
                  onClick={() => updateSetting('theme', t)}
                >
                  {t === 'dark' ? '🌙 Dark' : '☀️ Light'}
                </button>
              ))}
            </div>
          </SettingRow>

          <SettingRow label="Accent Color" description="Primary UI accent color">
            <div className="color-picker">
              {ACCENT_COLORS.map((c) => (
                <button
                  key={c.value}
                  className={`color-picker__swatch ${settings.accentColor === c.value ? 'color-picker__swatch--active' : ''}`}
                  style={{ background: c.value }}
                  onClick={() => updateSetting('accentColor', c.value)}
                  title={c.label}
                />
              ))}
            </div>
          </SettingRow>

          <SettingRow label="Animations" description="Enable smooth transitions and hover effects">
            <Toggle
              checked={settings.animationsEnabled}
              onChange={(v) => updateSetting('animationsEnabled', v)}
            />
          </SettingRow>
        </div>

        {/* Typography */}
        <div className="settings-group">
          <h3 className="settings-group__title"><Type size={18} /> Typography</h3>

          <SettingRow label="Font Size" description="Base font size for the interface">
            <div className="filter-bar__chips">
              {['small', 'medium', 'large'].map((s) => (
                <button
                  key={s}
                  className={`chip ${settings.fontSize === s ? 'chip--active' : ''}`}
                  onClick={() => updateSetting('fontSize', s)}
                >
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </button>
              ))}
            </div>
          </SettingRow>
        </div>

        {/* Data Display */}
        <div className="settings-group">
          <h3 className="settings-group__title"><Eye size={18} /> Data Display</h3>

          <SettingRow label="Confidence Intervals" description="Show 95% bootstrap CIs on charts">
            <Toggle
              checked={settings.showConfidenceIntervals}
              onChange={(v) => updateSetting('showConfidenceIntervals', v)}
            />
          </SettingRow>

          <SettingRow label="Chart Grid" description="Show grid lines on charts">
            <Toggle
              checked={settings.showGrid}
              onChange={(v) => updateSetting('showGrid', v)}
            />
          </SettingRow>

          <SettingRow label="Compact Tables" description="Reduce row padding in tables">
            <Toggle
              checked={settings.compactTables}
              onChange={(v) => updateSetting('compactTables', v)}
            />
          </SettingRow>

          <SettingRow label="Decimal Places" description="Number of decimal places for metrics">
            <div className="filter-bar__chips">
              {[1, 2, 3, 4].map((n) => (
                <button
                  key={n}
                  className={`chip ${settings.decimalPlaces === n ? 'chip--active' : ''}`}
                  onClick={() => updateSetting('decimalPlaces', n)}
                >
                  {n}
                </button>
              ))}
            </div>
          </SettingRow>
        </div>

        {/* Chart Options */}
        <div className="settings-group">
          <h3 className="settings-group__title"><BarChart3 size={18} /> Chart Options</h3>

          <SettingRow label="Smooth Curves" description="Use smooth interpolation for line charts">
            <Toggle
              checked={settings.chartSmoothing}
              onChange={(v) => updateSetting('chartSmoothing', v)}
            />
          </SettingRow>

          <SettingRow label="Default Metric" description="Default metric for chart tooltips">
            <div className="filter-bar__chips">
              {[
                { id: 'delta_psnr', label: 'ΔPSNR' },
                { id: 'ssim', label: 'SSIM' },
              ].map((m) => (
                <button
                  key={m.id}
                  className={`chip ${settings.defaultMetric === m.id ? 'chip--active' : ''}`}
                  onClick={() => updateSetting('defaultMetric', m.id)}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </SettingRow>
        </div>
      </div>
    </div>
  );
}
