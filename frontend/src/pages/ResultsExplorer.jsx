import { useState, useMemo } from 'react';
import {
  SUMMARY_BAND, DO_NO_HARM, METHODS, getMethodColor,
} from '../data/experimentData';
import { useSettings } from '../context/SettingsContext';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, ErrorBar,
} from 'recharts';
import { Filter, Download, ArrowUpDown, Table2 } from 'lucide-react';

export default function ResultsExplorer() {
  const { settings } = useSettings();
  const dp = settings.decimalPlaces;
  const [selectedMethods, setSelectedMethods] = useState(
    METHODS.map((m) => m.id)
  );
  const [bandFilter, setBandFilter] = useState('all');
  const [sortBy, setSortBy] = useState('mean');
  const [sortDir, setSortDir] = useState('desc');
  const [view, setView] = useState('table');

  const toggleMethod = (id) => {
    setSelectedMethods((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const filteredData = useMemo(() => {
    let data = SUMMARY_BAND.filter((d) => selectedMethods.includes(d.method));
    if (bandFilter !== 'all') data = data.filter((d) => d.band === bandFilter);
    data = [...data].sort((a, b) => {
      const mult = sortDir === 'asc' ? 1 : -1;
      return mult * (a[sortBy] - b[sortBy]);
    });
    return data;
  }, [selectedMethods, bandFilter, sortBy, sortDir]);

  const radarData = useMemo(() => {
    const metrics = ['In-Band ΔPSNR', 'Out-of-Band ΔPSNR', 'Do-No-Harm PSNR', 'Do-No-Harm SSIM'];
    return metrics.map((metric) => {
      const row = { metric };
      selectedMethods.forEach((mid) => {
        const m = METHODS.find((m) => m.id === mid);
        if (!m) return;
        if (metric === 'In-Band ΔPSNR') {
          const d = SUMMARY_BAND.find((s) => s.method === mid && s.band === 'in');
          row[m.name] = d ? Math.max(0, d.mean) : 0;
        } else if (metric === 'Out-of-Band ΔPSNR') {
          const d = SUMMARY_BAND.find((s) => s.method === mid && s.band === 'out');
          row[m.name] = d ? Math.max(0, d.mean) : 0;
        } else if (metric === 'Do-No-Harm PSNR') {
          const d = DO_NO_HARM.find((s) => s.method === mid);
          row[m.name] = d ? d.psnr / 5 : 0;
        } else {
          const d = DO_NO_HARM.find((s) => s.method === mid);
          row[m.name] = d ? d.ssim * 20 : 0;
        }
      });
      return row;
    });
  }, [selectedMethods]);

  const handleSort = (col) => {
    if (sortBy === col) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(col);
      setSortDir('desc');
    }
  };

  const exportCSV = () => {
    const headers = ['Method', 'Band', 'Mean ΔPSNR', '95% CI Lo', '95% CI Hi', 'N'];
    const rows = filteredData.map((d) => [
      d.method, d.band, d.mean, d.ci_lo, d.ci_hi, d.n,
    ]);
    const csv = [headers, ...rows].map((r) => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'freqlab_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="page">
      <div className="page__header">
        <div>
          <h1 className="page__title">Results Explorer</h1>
          <p className="page__description">
            Interactive exploration of 34,300 test evaluations with filtering, sorting, and export
          </p>
        </div>
        <button className="btn btn--secondary" onClick={exportCSV}>
          <Download size={16} /> Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="filter-bar">
        <div className="filter-bar__section">
          <span className="filter-bar__label"><Filter size={14} /> Methods</span>
          <div className="filter-bar__chips">
            {METHODS.map((m) => (
              <button
                key={m.id}
                className={`chip ${selectedMethods.includes(m.id) ? 'chip--active' : ''}`}
                style={selectedMethods.includes(m.id) ? { '--chip-color': m.color } : {}}
                onClick={() => toggleMethod(m.id)}
              >
                <span className="chip__dot" style={{ background: m.color }}></span>
                {m.name}
              </button>
            ))}
          </div>
        </div>
        <div className="filter-bar__section">
          <span className="filter-bar__label">Band</span>
          <div className="filter-bar__chips">
            {['all', 'in', 'out'].map((b) => (
              <button
                key={b}
                className={`chip ${bandFilter === b ? 'chip--active' : ''}`}
                onClick={() => setBandFilter(b)}
              >
                {b === 'all' ? 'All Bands' : b === 'in' ? 'In-Band' : 'Out-of-Band'}
              </button>
            ))}
          </div>
        </div>
        <div className="filter-bar__section">
          <span className="filter-bar__label">View</span>
          <div className="filter-bar__chips">
            <button className={`chip ${view === 'table' ? 'chip--active' : ''}`} onClick={() => setView('table')}>
              <Table2 size={14} /> Table
            </button>
            <button className={`chip ${view === 'chart' ? 'chip--active' : ''}`} onClick={() => setView('chart')}>
              <ArrowUpDown size={14} /> Chart
            </button>
            <button className={`chip ${view === 'radar' ? 'chip--active' : ''}`} onClick={() => setView('radar')}>
              🎯 Radar
            </button>
          </div>
        </div>
      </div>

      {/* Table View */}
      {view === 'table' && (
        <div className="chart-container">
          <table className={`data-table ${settings.compactTables ? 'data-table--compact' : ''}`}>
            <thead>
              <tr>
                <th onClick={() => handleSort('method')} className="sortable">
                  Method <ArrowUpDown size={12} />
                </th>
                <th>Type</th>
                <th>Band</th>
                <th onClick={() => handleSort('mean')} className="sortable">
                  Mean ΔPSNR <ArrowUpDown size={12} />
                </th>
                <th>95% CI</th>
                <th>CI Width</th>
                <th>N</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.map((row, i) => {
                const m = METHODS.find((m) => m.id === row.method);
                return (
                  <tr key={i}>
                    <td>
                      <span className="method-dot" style={{ background: m?.color }}></span>
                      {m?.name || row.method}
                    </td>
                    <td><span className={`type-badge type-badge--${m?.type}`}>{m?.type}</span></td>
                    <td>
                      <span className={`band-badge band-badge--${row.band}`}>
                        {row.band === 'in' ? 'In-Band' : 'Out-of-Band'}
                      </span>
                    </td>
                    <td className="mono">
                      <span className={row.mean > 0 ? 'positive' : row.mean < 0 ? 'negative' : ''}>
                        {row.mean > 0 ? '+' : ''}{row.mean.toFixed(dp)}
                      </span>
                    </td>
                    <td className="mono">[{row.ci_lo.toFixed(dp)}, {row.ci_hi.toFixed(dp)}]</td>
                    <td className="mono">{(row.ci_hi - row.ci_lo).toFixed(dp)}</td>
                    <td className="mono">{row.n}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="table-footer">
            Showing {filteredData.length} of {SUMMARY_BAND.length} entries
          </div>
        </div>
      )}

      {/* Chart View */}
      {view === 'chart' && (
        <div className="chart-container">
          <h3 className="chart-container__title">ΔPSNR by Method & Band</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={filteredData.map((d) => ({
              name: `${METHODS.find((m) => m.id === d.method)?.name} (${d.band})`,
              ΔPSNR: d.mean,
              color: getMethodColor(d.method),
              err: [d.mean - d.ci_lo, d.ci_hi - d.mean],
            }))} layout="vertical" barGap={2}>
              {settings.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />}
              <XAxis
                type="number"
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
                label={{ value: 'ΔPSNR (dB)', position: 'bottom', fill: 'var(--text-secondary)' }}
              />
              <YAxis
                dataKey="name"
                type="category"
                width={150}
                tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--border)' }}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)',
                }}
              />
              <Bar dataKey="ΔPSNR" radius={[0, 4, 4, 0]}>
                {filteredData.map((d, i) => (
                  <Cell key={i} fill={getMethodColor(d.method)} />
                ))}
                {settings.showConfidenceIntervals && (
                  <ErrorBar dataKey="err" width={4} strokeWidth={1.5} stroke="var(--text-secondary)" />
                )}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Radar View */}
      {view === 'radar' && (
        <div className="chart-container">
          <h3 className="chart-container__title">Multi-Metric Radar Comparison</h3>
          <ResponsiveContainer width="100%" height={450}>
            <RadarChart data={radarData} cx="50%" cy="50%">
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
              <PolarRadiusAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }} />
              {selectedMethods.map((mid) => {
                const m = METHODS.find((mm) => mm.id === mid);
                if (!m) return null;
                return (
                  <Radar
                    key={mid}
                    name={m.name}
                    dataKey={m.name}
                    stroke={m.color}
                    fill={m.color}
                    fillOpacity={0.15}
                    strokeWidth={2}
                  />
                );
              })}
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
          <p className="chart-container__note">
            PSNR values scaled by ÷5 and SSIM by ×20 for visual comparability on the radar axes.
          </p>
        </div>
      )}
    </div>
  );
}
