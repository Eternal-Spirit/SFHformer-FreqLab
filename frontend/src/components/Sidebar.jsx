import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Wand2,
  Briefcase,
  BarChart3,
  LineChart,
  FlaskConical,
  Settings,
  Microscope,
  ChevronLeft,
  ChevronRight,
  Eye,
} from 'lucide-react';
import { useState } from 'react';

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/restore', icon: Wand2, label: 'Restore Studio' },
  { to: '/retina', icon: Eye, label: 'RetinaScan Studio' },
  { to: '/use-cases', icon: Briefcase, label: 'Use Cases' },
  { to: '/results', icon: BarChart3, label: 'Results Explorer' },
  { to: '/visualizations', icon: LineChart, label: 'Visualizations' },
  { to: '/training', icon: FlaskConical, label: 'Training Monitor' },
  { to: '/methodology', icon: Microscope, label: 'Methodology' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
      <div className="sidebar__header">
        {!collapsed && (
          <div className="sidebar__brand">
            <div className="sidebar__logo">
              <span className="sidebar__logo-icon">⚡</span>
            </div>
            <div className="sidebar__brand-text">
              <span className="sidebar__title">FreqLab</span>
              <span className="sidebar__subtitle">SFHformer</span>
            </div>
          </div>
        )}
        <button
          className="sidebar__toggle"
          onClick={() => setCollapsed(!collapsed)}
          aria-label="Toggle sidebar"
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      <nav className="sidebar__nav">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`
            }
            title={label}
          >
            <Icon size={20} className="sidebar__link-icon" />
            {!collapsed && <span className="sidebar__link-label">{label}</span>}
          </NavLink>
        ))}
      </nav>

      {!collapsed && (
        <div className="sidebar__footer">
          <div className="sidebar__footer-badge">
            <span className="sidebar__footer-dot"></span>
            ECCV 2024
          </div>
          <div className="sidebar__footer-info">v1.0 · Completed 2026-10-26</div>
        </div>
      )}
    </aside>
  );
}
