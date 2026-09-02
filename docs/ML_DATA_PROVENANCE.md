# SkyGuard AI — ML Data Provenance

Complete record of every dataset, artifact, seed, hash, split and experiment log
that backs the SkyGuard v2 production model and the v3 improvement campaign.
Everything here is machine-verified; nothing is claimed beyond what files/artifacts
record. **Date:** 2026-08-29.

---

## 1. Provenance principles

- **Trained once, offline, frozen.** No online retraining or adaptation at
  inference. Artifacts are persisted to disk and loaded, not recomputed.
- **Hash-pinned inputs.** Any dataset mismatch halts training.
- **Chronological splits, never shuffled.** No future-data leakage into
  training; validation and test windows are later than training.
- **Tuning never touches evaluation data.** Validation-only search; benchmark
  (seed 7) and unseen (seed 19) frames excluded by construction.
- **Synthetic disclosed.** Every dataset is synthetic-but-weather-typical;
  `synthetic: true` is visible in metadata and UI.
- **Scores uncalibrated.** Detector outputs are evidence, not probabilities.

## 2. Production dataset — `sample_weather_v2`

| Property | Value |
|---|---|
| File | `data/sample_weather.csv` |
| SHA-256 | `69b56c7757035ee64bf979fede8499e94c7968c3004cd16037a37baede17b4fe` |
| Generator | station-distinct v2 telemetry (lat/elevation/distinct diurnal/seeded) |
| Stations / rows / features | 24 / 32,256 / 25 |
| Variables | temperature, pressure, humidity |
| Split (chronological, never shuffled) | train 70% · val 15% · test 15% |
| Synthetic | true (disclosed) |
| Artifact | `backend/models/skyguard_v2/` (frozen; `metadata.json`) |

## 3. Campaign dataset — `sample_weather_v3`

| Property | Value |
|---|---|
| File | `data/sample_weather_v3.csv` (+ `.meta.json`) |
| SHA-256 | `7d20d2d1a990813087fd712499905e27e0f82d5e1ef8ab93c2dc39a08d618511` |
| Generator | `backend/data_generator_v3.py` |
| Seeds | generation `91` · regime `4242` · episode `777` |
| Stations / rows | 24 stations × 60 days × 96 (15-min) = **138,240** observations, 5,760/station |
| Period | 2026-06-01 00:00 → 2026-07-30 23:45 |
| Variables | temperature, pressure, humidity (25 computed features) |
| Regimes | warm/cool phase · dry/humid phase · pressure rise/fall · episodic rain |
| Synthetic | true (disclosed) |
| Anomaly types (injected via eval framework) | SPIKE (dur 1) · DROP (dur 1) · DRIFT (dur 8) · FROZEN_SENSOR (dur 6) · MISSING_DATA (dur 5) |

Split (persisted, chronological, never shuffled):

| Split | Rows | Boundaries (incl.) |
|---|---|---|
| TRAIN | 96,768 | 2026-06-01 00:00 → 2026-07-12 23:45 |
| VALIDATION | 20,736 | 2026-07-13 00:00 → 2026-07-21 23:45 |
| TEST | 20,736 | 2026-07-22 00:00 → 2026-07-30 23:45 |

## 4. Model artifacts

### `backend/models/skyguard_v2/` — PRODUCTION (unchanged)

`skyguard-multivariate-v2` · dataset `sample_weather_v2` · threshold **0.72** ·
synthetic true · uncalibrated · ml:model_fusion + rule/statistical components.
File mtimes untouched by the campaign (v2 artifact written 2026-08-29 16:07).

### `backend/models/skyguard_v3/` — EXPERIMENTAL (frozen, not promoted)

`skyguard-multivariate-v3` · dataset `sample_weather_v3` · threshold **0.30** ·
fusion weights ml 0.30 / drift 0.10 / flat 0.30 / miss 0.30 · synthetic true ·
uncalibrated · trained once offline. Contents: `model.pkl`, `metadata.json`,
`feature_schema.json`.

ML channel hyperparameters (`metadata.json`): Isolation Forest — baseline
(n_est 100, contam 0.05, rs 42), temporal (contam 0.04, rs 7), multivariate
(contam 0.04, rs 21); seasonal = diurnal-profile residuals. Scalers fit on
TRAIN only. Mechanism channels (config-only, no data fit): drift
(ewma_alpha 0.4, cusum k 0.55, h 3.0, slope win 8), flatline (window 5,
ratio_floor 0.4), missingness (interval_tol_fraction 1.5, window 12).

## 5. Evaluation frames (generated on demand, never trained on)

Built by the unchanged `services._build_eval_frame(rows, seed)` generator
(isolated realistic ground truth; ground truth kept separate from predictions).

| Frame | Seed | Rows | Anomalies | Used for |
|---|---|---|---|---|
| Validation | 11 | 600 | 138 (rate 0.23) | Phase 2 tuning (ONLY frame) |
| Benchmark | 7 | 1,300 | 144 (rate 0.1108) | Phase 3 evaluation (v1/v2/v3) |
| Unseen | 19 | 1,300 | 144 (rate 0.1108) | Phase 3 evaluation (v2 vs v3) |

Anomaly types per frame: SPIKE / DROP / DRIFT / FROZEN_SENSOR / MISSING_DATA
with durations above.

## 6. Experiment logs (every candidate preserved)

| File | Run |
|---|---|
| `ml/experiments/v3_val_tuning_seed11_20260829T133248Z.json` | first pass, grid floor 0.40–0.775 (boundary artifact identified) |
| `ml/experiments/v3_val_tuning_seed11_20260829T133636Z.json` | intermediate |
| `ml/experiments/v3_val_tuning_seed11_20260829T134709771558Z.json` | intermediate |
| `ml/experiments/v3_val_tuning_seed11_20260829T134905934529Z.json` | **selected run (frozen config)** |
| `ml/experiments/v3_val_tuning_seed11_20260829T135548859774Z.json` | determinism double-run (identical search) |
| `ml/experiments/v3_val_tuning_seed11_latest.json` | pointer to latest |

- 11,700 candidates per run (5 detector configs × 30 weight combos × 78 thresholds).
- Determinism: second run produced identical selected config and candidate set;
  only run-id/timestamp fields differ.
- Persisted Phase 3 results: `ml/evaluation/phase3_seed7_versions.json`,
  `ml/evaluation/phase3_seed19_v2v3.json`,
  `ml/experiments/v3_campaign_final_summary.json`.

## 7. Timeline (2026-08-29, local/UTC)

| UTC | Event |
|---|---|
| 10:37 | v2 artifact trained (production; in metadata of live server) |
| 13:32–13:55 | v3 tuning runs (5 experiment logs) |
| 13:49 | v3 artifact trained (run `20260829T134905934529Z`), persisted to disk |
| 13:55 | determinism double-run |
| after 13:55 | Phase 3 read-only evaluation: seed 7 (v1/v2/v3) + seed 19 (v2/v3) |
| 19:xx | Option A closure: regression tests, demo smoke, reports |

## 8. Honesty & limitations

- All datasets are synthetic; `synthetic: true` is disclosed in artifact
  metadata and the UI.
- Scores are uncalibrated fusion output, never probabilities (`calibration:
  none` in metadata for both artifacts).
- v3 is recorded as an experimental artifact; the production decision
  (keep v2) is recorded in `docs/ML_Performance_Improvement_Report.md` §2/§7
  and `ml/experiments/v3_campaign_final_summary.json`.
- v2 FROZEN/MISSING recall is 0.0000 on the benchmark; this is an open,
  disclosed limitation of the deployed model.
- No claim of verified-real-weather reasoning is made anywhere; the
  weather-vs-sensor heuristic is spatial evidence only.