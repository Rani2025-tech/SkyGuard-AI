# SkyGuard AI — SIH Demo Runbook

**What:** 8-minute live demo of SkyGuard AI (multi-station weather anomaly detection &
self-correction) for the Smart India Hackathon judging panel.
**Built on:** Anomaly detection with fused ML + rule/stats, anomaly-cause attribution,
self-correction, honest provenance, cross-station weather-vs-sensor reasoning,
model lineage + evaluation.

---

## 0. Pre-flight checklist (do this BEFORE the judges arrive)

```powershell
# 1. Start backend (port 8000)
cd C:\sih_demo\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2. Wait for "Application startup complete" (pipeline trains ~5-15 s),
#    then confirm:  http://127.0.0.1:8000/api/health  -> "online", model_trained: true

# 3. Start frontend (port 5173)
cd C:\sih_demo\frontend
npm run dev
#    Open http://localhost:5173
```

- Restart order: backend first, frontend second.
- If you want a clean demo dataset: stop backend, delete `backend\skyguard.db`,
  restart backend (DB is recreated on startup).
- Keep a terminal visible for the Evaluation page call (≈4.6 s) — it recomputes on
  cache TTL; treat that as a feature, not a glitch.
- Verified today against the frozen artifacts (see §8 rehearsal log).

## 1. 60-second pitch (memorize)

> SkyGuard is an operational weather-anomaly detection system for a 24-station AWS
> (automatic weather station) network. Every station streams temperature, pressure
> and relative humidity at 5-minute resolution. SkyGuard watches for sensor faults,
> drifts and extreme events; when something looks wrong it tells you **what**, **how
> confident it is**, **why the algorithm thinks so**, **whether it is a sensor fault
> or real weather**, and it **proposes a cleaned value** — with every claim labeled
> by which component (ML, rule, or statistical) produced it. The whole thing is
> trained once on a versioned, hash-pinned dataset; nothing is re-trained at
> inference time, so results are reproducible.

## 2. What the demo is (and honestly is NOT)

- We simulate anomalies **on a copy** of a station's recent stream; ground truth is
  the injected anomaly, kept separate from the model's prediction.
- The anomaly score is **uncalibrated** — it is fused evidence output, not a
  probability. The UI says so.
- Dataset is synthetic (weather-typical, seeded), synthetic flag is visible in model
  metadata — we disclose it, never hide it.

## 3. Demo flow (≈8 min)

### Segment A — Dashboard overview (90 s)
- Home → Dashboard: healthy stream, station grid, recent alerts, "What SkyGuard does"
  legend (ML · Rule · Statistical), score shown as a **raw uncalibrated score**, not %.
- Say: "Live status across 24 stations; metric cards show the model lineage, and the
  legend maps each number to the component that produced it."

### Segment B — Simulation: 3 scenarios (3 min)
Go to **Simulation**. Three exact injections below. All verified to detect.

#### Scenario 1 — Temperature SPIKE (sensor glitch)
| Field | Value |
|---|---|
| Station / variable | AWS-023 / temperature |
| Anomaly type | SPIKE |
| Magnitude / duration | 14 σ / 1 |
| Result | **Detected → HIGH**, type SPIKE |
| ML score | **0.844** (raw, uncalibrated) |
| Rule classifier | SPIKE |
| Fault likelihood | 0.13 (statistical heuristic) |
| Assessment | **LIKELY_SENSOR_FAULT** |

What to say: "A single reading jumps ~14 standard deviations. The fused ML model
flags it; the rule classifier names it a spike; the statistical heuristic reasons it
is sensor-side, not weather." Then open **Anomaly Detail** (bottom-right of swarm
cleaner) →
- **Attribution (Model Attribution — Counterfactual Occlusion):** temperature
  contribution **+0.1928** — the real model re-scored after replacing the spike with
  the typical value.
- **Type source:** rule/classifier. **Score:** ml:model_fusion.
- **Correction:** statistical diurnal estimate (`recent-window mean`), corrected value
  shown with the original crossed out.

#### Scenario 2 — Humidity DRIFT (sensor degradation)
| Field | Value |
|---|---|
| Station / variable | AWS-023 / relative humidity |
| Anomaly type | DRIFT |
| Magnitude / duration | 15 σ / 6 |
| Result | **Detected**, type DRIFT |
| ML score | **0.798** |

Say: "Gradual drift over 30 minutes — the sensor drifts rather than fails. The
rule classifier reads the trend from the series (DRIFT), the ML detects it, and the
statistical resilience metric (drift vs stability) contributes to the health score."

#### Scenario 3 — Cross-station consistency / weather vs sensor (2 min)
Same SPIKE injected on the three **adjacent** stations (adjacency verified from
`data_generator_meta`: AWS-023 ↔ AWS-002 ↔ AWS-003):

| Station | Result | ML score |
|---|---|---|
| AWS-023 | detected | 0.844 |
| AWS-002 | detected | 0.845 |
| AWS-003 | detected | 0.851 |

What to say — **this is the honest teaching moment**:
> "Here we demonstrate the spatial consistency check. Because our simulation engine
> injects each station in isolation — the neighbours' stored stream stays normal —
> the Weather-vs-Sensor heuristic sees one hot station among 24 and correctly leans
> *sensor side* (likelihood 0.13, 'partial consensus'). If several neighbouring
> stations deviated in the same direction simultaneously — a real regional weather
> event — the same evidence lines shift the assessment toward *weather*. The model
> exposes those evidence lines on the detail screen; we don't hide the ambiguity."

Never claim the UI shows `weather` for a simulation — it will not with
isolated injection.

> Skip note: sharp DROP and trailing FROZEN scenarios are tuning-sensitive with the
> current detector (borderline detect / read as DRIFT). We use SPIKE + DRIFT live.

### Segment B2 — Live Monitor deterministic demo stream (~90 s)
The **Live Monitor** screen replays a fixed 35-reading AWS-023 episode. It shows
`🚨 ANOMALY DETECTED` at three predictable reading numbers, with Normal readings
in between, in the same order on every run:

| Reading # | Station | Type | Feature | ML score | Severity |
|---|---|---|---|---|---|
| **13** | AWS-023 | SPIKE | temperature | 0.844 | HIGH |
| **23** | AWS-023 | DRIFT | humidity | 0.798 | HIGH |
| **33** | AWS-023 | FROZEN_SENSOR | temperature | 0.855 | HIGH |

- These three events reuse real, already-detected anomaly *records* of the same
  **station (`AWS-023`) and same type**, so the Anomalies page shows a
  corresponding record for each (one per representative type).
- **Honest match note (tell this to judges):** the correspondence is
  **type-based, not an exact timestamp match.** The Live Monitor row displays the
  timestamp of the *replayed source-data reading* (a historical `sample_weather`
  timestamp, e.g. `2026-08-14 13:00:00` for reading 13). The matching Anomalies-page
  record carries its own DB-timestamp (the time the detection was recorded/seeded).
  Station + anomaly type are identical; the timestamps are **not** identical. The
  UI marks these rows as **Demo** for exactly this reason.
- This keeps the module "reuse existing records / do not change detection logic":
  no new model is trained and the v3 detector is unchanged.

### Segment C — Correction & human-in-the-loop (60 s)
- Same screen: corrected value, correction method + provenance, manual override path.
- Switch to **Station Detail**: raw uncalibrated score, health factors (drift /
  stability), status timeline.

### Segment D — Model, Evaluation, Settings (90 s)
- **Model page**: `skyguard-multivariate-v3`, dataset `sample_weather_v3`,
  sha256 `7d20d2d1…`, 24 stations / 138,240 observations (60 days @ 15-min) /
  25 features, threshold 0.3, calibration **uncalibrated**, synthetic **true**,
  artifact persisted on disk (`backend/models/skyguard_v3/`).
- **Evaluation page**: labeled blocks —
  - **Production pipeline (v3)**: precision 0.8485 · recall 0.5833 · F1 0.6914 ·
    ROC-AUC 0.8994 (n = 1,300).
  - **Version comparison (measured)**: V3 = **CURRENT PRODUCTION**,
    V2 = LEGACY (prior production), V1 = legacy artifact, each scored on the
    identical hard injected frame at its own artifact threshold.
  - **Legacy artifacts**: v2 precision 0.9483 · recall 0.3819 · F1 0.5446 ·
    ROC-AUC 0.7506; v1 precision 0.1229 · recall 0.7986 · ROC-AUC 0.6606.
  - **Ablation (research)** + per-type recall; note the honest difference caption:
    the ablation chart uses the standalone analysers, the production cards use the
    fused pipeline — different paths, labeled as such (ablation FINAL 0.7558 vs
    current production v3 0.8485).
- **Settings page**: accurate attribution wording (no "SHAP-style z-score" claims).

## 4. Honesty guardrails (never assert)

- ❌ "The score is a probability" — it is uncalibrated raw fusion output.
- ❌ "The weather assessment is verified against real weather" — the spatial
  heuristic exists; simulations read sensor-leaning.
- ❌ "The model detects FROZEN/MISSING perfectly" — v3 recall is 0.50 / 0.70
  (real, not perfect); v2 was 0.0 / 0.0. Say exactly that.
- ❌ "v3 simply beats v2 on everything" — the trade-off is real: v3 lifts overall
  recall, F1, ROC-AUC and FROZEN/MISSING detection, but precision drops
  (0.9483 → 0.8485) and SPIKE/DROP recall falls vs v2. Say exactly that.
- ❌ "Retraining happens at inference" — false; model is trained once, frozen.

## 5. Judge Q&A bank

| Question | Answer (all verified today) |
|---|---|
| What is the core problem? | 24-station AWS network; operators drown in false alarms and miss sensor degradation. |
| What does SkyGuard detect? | SPIKE / DROP / DRIFT / SUDDEN_SHIFT level shifts / frozen sensors / missing data — with a severity level. |
| Where is ML vs rules vs stats? | ML: fused anomaly score (weighted evidence fusion over isolation-forest detectors: baseline, temporal, seasonal, multivariate + drift/flatline/missingness channels). Rules: type classifier (drift slope, plateau, shift tests). Stats: z-scores, sensor-fault likelihood, correction estimates. Every field on screen is labeled (`computed_by`). |
| How does detection actually work? | 25 features (levels, rolling means, volatility, cross-variable pressure/humidity deviation); detectors vote; fusion yields the uncalibrated score; production threshold 0.3 on the fused score raises the alert. |
| How do you tell weather from a sensor fault? | Weather-vs-sensor heuristic: crosses with same-hour diurnal/seasonal expectation + spatial neighbours + multi-variable co-movement; evidence lines are exposed on the anomaly detail. |
| Can you prove attributions? | Model-true counterfactual occlusion: replace the anomalous value with the typical → re-score through the real model → temperature contributed +0.1928. Method is labeled `counterfactual` in the API/DB. |
| Why is v3 the current production model? | v3 lifts overall recall, F1, ROC-AUC and FROZEN/MISSING detection over v2 (F1 0.5446 → 0.6914, ROC-AUC 0.7506 → 0.8994) at the cost of precision (0.9483 → 0.8485) and SPIKE/DROP recall; v2 keeps the precision edge. Both v2 and v1 remain legacy artifacts. |
| Is the data real? | No. `sample_weather_v3` is synthetic but weather-typical, seeded, and hash-pinned (sha256 `7d20d2d1…`, 138,240 obs / 24 stations / 60 days); synthetic flag is displayed, not hidden. Production would swap in the live stream via the ingestion layer. |
| How reproducible? | Model trained once from the pinned dataset; production artifact persisted (`backend/models/skyguard_v3/`), v2/v1 kept as legacy artifacts; evaluation rebuilds an isolated realistic ground truth; no online retraining. |
| What is the latency? | Injection → result ~0.2-0.3 s locally (measured: 0.17-0.33 s); Evaluation page full recompute ≈4.6 s (isolated job). |
| What would you ship next? | Calibrated probabilities (blocked on a validated calibration split — deliberately not enabled), adaptive per-station thresholds, and the production data stream. |

## 6. Quantitative evidence sheet

| Metric | v3 (current production) | v2 (legacy) | v1 (legacy) |
|---|---|---|---|
| Model version | `skyguard-multivariate-v3` | `skyguard-multivariate-v2` | v1 artifact |
| Dataset | `sample_weather_v3` (sha `7d20d2d1…`) | `sample_weather_v2` (sha `69b56c7757035ee6…`) | v1 |
| Test windows (n) | 1,300 | 1,300 | 1,300 |
| Precision | 0.8485 | **0.9483** | 0.1229 |
| Recall | 0.5833 | 0.3819 | **0.7986** |
| F1 | **0.6914** | 0.5446 | 0.2130 |
| ROC-AUC | **0.8994** | 0.7506 | 0.6606 |

- Per-type recall (same frame): **v3 (current production)** SPIKE 0.33 · DROP 0.40 ·
  DRIFT 0.71 · FROZEN_SENSOR 0.50 · MISSING_DATA 0.70;
  **v2 (legacy)** SPIKE 0.80 · DROP 0.67 · DRIFT 0.69 · FROZEN_SENSOR 0.0 ·
  MISSING_DATA 0.0.
- Ablation (standalone analysers, **different path** than production fusion): Baseline
  AUC 0.7512 · +Temporal 0.7517 · +Diurnal/Seasonal 0.8916 · Multivariate 0.7519;
  +Diurnal/Seasonal cuts false alarms most (precision 0.6283 → 0.7941).
- Provenance: anomaly score = `ml:model_fusion`; anomaly type = `rule/classifier`;
  event assessment + fault likelihood + corrected value = `statistical heuristic`.
- Verification of distribution drift: neural density estimator + G-test, alpha 0.10
  (statistics page).

## 7. Runbook notes

- Two uvicorn processes were found holding port 8000 from earlier sessions (pre-PHASE-4
  code). They were stopped (PID 15144, 16524). Start ONE fresh server only; a stale
  server silently serves old responses (missing `method`, `sensor_fault_likelihood`).
- Fix shipped during rehearsal: `schemas.ExplanationOut` now declares `method`
  (FastAPI `response_model` stripped it before — frontend never received it). DB row
  was always correct; only the HTTP envelope dropped the field. Backend suite: 29/29 OK.

## 8. Rehearsal log (this session)

| Check | Result |
|---|---|
| GET /api/health | online · model_trained true |
| S1 SPIKE AWS-023 t mag14 | detected, HIGH, score 0.844, type SPIKE, likelihood 0.1303, assess LIKELY_SENSOR_FAULT, latency 333 ms |
| GET /api/anomalies/325 | score 0.844, `sensor_fault_likelihood` + `computed_by` populated |
| GET /api/explanations/325 | method `counterfactual`, temperature +0.1928 |
| S2 DRIFT hum mag15 dur6 | detected, score 0.798, latency 290 ms |
| S3 cross-station SPIKE | AWS-002 0.845 · AWS-003 0.851 (both detected) |
| GET /api/model/metadata | skyguard-multivariate-v3 · sample_weather_v3 · sha 7d20d2d1a99081 · 24/138240/25 · threshold 0.3 · uncalibrated · synthetic |
| GET /api/model/evaluation | detailed n=1,300 · 0.8485/0.5833/0.6914/0.8994; versions v3 = current production vs v2/v1 = legacy artifacts |
| GET /api/stations/AWS-023/predict | NORMAL score 0.3931 · source HISTORICAL |

Persisted demo anomaly IDs in `backend\skyguard.db`: 321–328 (rehearsal); 325 (SPIKE)
and 326 (DRIFT) are the reference rows with full explanations.