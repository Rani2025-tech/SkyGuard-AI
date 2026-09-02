import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Live from './pages/Live';
import Stations from './pages/Stations';
import StationDetail from './pages/StationDetail';
import Anomalies from './pages/Anomalies';
import AnomalyDetail from './pages/AnomalyDetail';
import Analytics from './pages/Analytics';
import SensorHealth from './pages/SensorHealth';
import XaiDeepDive from './pages/XaiDeepDive';
import Simulation from './pages/Simulation';
import Model from './pages/Model';
import Evaluation from './pages/Evaluation';
import Settings from './pages/Settings';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/live" element={<Live />} />
        <Route path="/stations" element={<Stations />} />
        <Route path="/stations/:id" element={<StationDetail />} />
        <Route path="/anomalies" element={<Anomalies />} />
        <Route path="/anomalies/:id" element={<AnomalyDetail />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/deep-dive" element={<XaiDeepDive />} />
        <Route path="/sensor-health" element={<SensorHealth />} />
        <Route path="/simulation" element={<Simulation />} />
        <Route path="/model" element={<Model />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  );
}
