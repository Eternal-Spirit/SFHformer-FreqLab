import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SettingsProvider } from './context/SettingsContext';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import ImageRestoration from './pages/ImageRestoration';
import UseCases from './pages/UseCases';
import ResultsExplorer from './pages/ResultsExplorer';
import Visualizations from './pages/Visualizations';
import TrainingMonitor from './pages/TrainingMonitor';
import Methodology from './pages/Methodology';
import SettingsPage from './pages/SettingsPage';
import RetinaStudio from './pages/RetinaStudio';
import './index.css';

export default function App() {
  return (
    <SettingsProvider>
      <BrowserRouter>
        <div className="app">
          <Sidebar />
          <main className="main">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/restore" element={<ImageRestoration />} />
              <Route path="/retina" element={<RetinaStudio />} />
              <Route path="/use-cases" element={<UseCases />} />
              <Route path="/results" element={<ResultsExplorer />} />
              <Route path="/visualizations" element={<Visualizations />} />
              <Route path="/training" element={<TrainingMonitor />} />
              <Route path="/methodology" element={<Methodology />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </SettingsProvider>
  );
}
