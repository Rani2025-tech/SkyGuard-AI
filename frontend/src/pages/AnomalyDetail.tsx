import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api';
import type { Anomaly } from '../types';
import ExplanationPanel from '../components/ExplanationPanel';
import { PageHeader, Card, Badge, Spinner, ErrorState, fmt } from '../components/ui';
import { IconArrowLeft, IconAlert, IconCheck, IconInfo } from '../components/icons';

function Row({ label, value, mono }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2.5 border-b border-borderline/50 last:border-0">
      <span className="text-[11px] text-slate-500 uppercase tracking-wider">{label}</span>
      <span className={`text-sm text-slate-100 ${mono ? 'font-mono text-[13px]' : ''}`}>{value}</span>
    </div>
  );
}

export default function AnomalyDetail() {
  const { id = '' } = useParams();
  const [anomaly, setAnomaly] = useState<Anomaly | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setAnomaly(null);
    setError(null);
    api
      .anomaly(Number(id))
      .then(setAnomaly)
      .catch((e) => setError(e.message));
  }, [id]);

  if (error) return <ErrorState message={error} />;
  if (!anomaly) return <Spinner label="Loading anomaly\u2026" />;

  return (
    <div className="space-y-7">
      <PageHeader
        title={`Anomaly #${anomaly.id}`}
        icon={<IconAlert size={20} />}
        subtitle={<Link to="/anomalies" className="inline-flex items-center gap-1.5 text-xs text-sky-400 hover:text-sky-300 font-medium transition-colors"><IconArrowLeft size={14} /> Back to anomalies</Link>}
        actions={
          <Badge status={anomaly.severity}>
            <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse-soft" /> {anomaly.severity} severity
          </Badge>
        }
      />

      <div className="grid lg:grid-cols-2 gap-6">
        <Card title="Detection Summary" subtitle={new Date(anomaly.timestamp).toLocaleString()}>
          <Row label="Station" value={<span className="font-semibold text-sky-300">{anomaly.station_id}</span>} />
          <Row label="Anomaly Type" value={<span className="capitalize">{anomaly.anomaly_type?.toLowerCase()}</span>} />
          <Row label="Affected Variable" value={anomaly.feature ?? '\u2014'} />
          <Row label="Observed Value" value={<span className="text-red-400 font-semibold">{fmt(anomaly.raw_value)}</span>} mono />
          <Row label="Expected Value" value={fmt(anomaly.expected_value)} mono />
          <Row
            label="Anomaly Score"
            value={
              <span title="Fused model score — not a calibrated probability">
                {anomaly.score != null ? fmt(anomaly.score, 3) : '\u2014'}
                <span className="ml-1.5 text-[10px] font-normal text-slate-500 uppercase">uncalibrated</span>
              </span>
            }
            mono
          />
          <Row
            label="Type Source"
            value={<span className="text-[11px] font-normal text-slate-400">{anomaly.anomaly_type_source ?? 'rule/classifier'}</span>}
          />
          <Row
            label="Assessment"
            value={
              anomaly.is_weather_event ? (
                <Badge status="healthy">Genuine weather</Badge>
              ) : (
                <span className="text-red-400 font-medium">LIKELY SENSOR FAULT</span>
              )
            }
          />
        </Card>

        <Card title="Corrected Value" subtitle="Statistical imputation (diurnal estimate / fallback) \u2014 not ML">
          {anomaly.corrected_value !== null && anomaly.corrected_value !== undefined ? (
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.06] p-4">
              <div className="flex items-center gap-2 text-emerald-300 mb-1">
                <IconCheck size={16} />
                <span className="text-[11px] font-semibold uppercase tracking-wider">Imputed reading</span>
              </div>
              <div className="text-4xl font-extrabold text-emerald-300 mb-1 tabular-nums">
                {fmt(anomaly.corrected_value)}
                {anomaly.feature === 'pressure' ? ' hPa' : anomaly.feature === 'humidity' ? ' %' : ' \u00b0C'}
              </div>
              <div className="text-xs text-slate-400 mb-4">Method: {anomaly.correction_method ?? '\u2014'}</div>
              <Row label="Correction Confidence" value={`${fmt((anomaly.correction_confidence ?? 0) * 100, 1)}%`} mono />
              <div className="mt-3 text-xs text-slate-500">
                Difference from observed: {fmt((anomaly.corrected_value ?? 0) - (anomaly.raw_value ?? 0), 1)}
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-500 py-12 text-center">No corrected value computed for this anomaly.</div>
          )}
        </Card>
      </div>

      {anomaly.event_assessment && (
        <Card title="Weather vs Sensor-Fault Assessment" subtitle="Spatial consistency across neighboring stations">
          <div className="flex flex-wrap items-center gap-4 mb-4">
            <Badge status={anomaly.is_weather_event ? 'normal' : 'critical'}>
              {anomaly.event_assessment.toLowerCase().replace(/_/g, ' ')}
            </Badge>
            {anomaly.event_assessment && (
              <span
                className="text-xs text-slate-400"
                title="Rule-based spatial heuristic, not a trained probability"
              >
                sensor-fault likelihood (rule heuristic) = {fmt(anomaly.sensor_fault_likelihood ?? 0, 2)}
              </span>
            )}
          </div>
          {anomaly.event_evidence ? (
            <ul className="text-xs text-slate-300 space-y-1.5 list-none">
              {anomaly.event_evidence.split('\n').filter(Boolean).map((line, i) => (
                <li key={i} className="flex items-start gap-2">
                  <IconInfo size={13} className="mt-0.5 shrink-0 text-slate-500" />
                  <span className="leading-relaxed">{line}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="text-xs text-slate-500">No detailed evidence recorded.</div>
          )}
        </Card>
      )}

      <ExplanationPanel anomalyId={anomaly.id} />
    </div>
  );
}