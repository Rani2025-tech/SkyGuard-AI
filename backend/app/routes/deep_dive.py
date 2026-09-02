"""deep_dive.py
------------
FastAPI endpoints for Multivariate Telemetry & XAI Deep-Dive,
Flagged Telemetry Anomalies Log, and Interactive SHAP Explainability Inspector.
Directly powered by the weather_anomaly_detection package.
"""

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse
import pandas as pd
import numpy as np
import io
import sys
import os
from typing import Optional, List, Dict, Any

# Ensure weather_anomaly_detection modules are importable
WAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../weather_anomaly_detection"))
if WAD_DIR not in sys.path:
    sys.path.insert(0, WAD_DIR)

from data_generator import generate_synthetic_weather_data
from anomaly_injector import inject_anomalies
from detector import WeatherAnomalyDetector
from explainer import AnomalyExplainer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

router = APIRouter(prefix="/deep-dive", tags=["deep-dive"])

# In-memory cache for processed datasets and explainers
_CACHE: Dict[str, Dict[str, Any]] = {}


def _unwrap_val(val, default):
    if val is None:
        return default
    if hasattr(val, "default"):
        return val.default
    return val


def _get_cache_key(station_id: str, days: Any, contamination: Any, seed: Any) -> str:
    station_clean = str(station_id).split()[0].upper()
    d = int(_unwrap_val(days, 30))
    c = float(_unwrap_val(contamination, 0.055))
    s = int(_unwrap_val(seed, 42))
    return f"{station_clean}_{d}_{c:.4f}_{s}"


def _get_or_create_processed_data(
    station_id: str = "AWS-1",
    days: Any = 30,
    contamination: Any = 0.055,
    seed: Any = 42
) -> Dict[str, Any]:
    days_val = int(_unwrap_val(days, 30))
    contam_val = float(_unwrap_val(contamination, 0.055))
    seed_val = int(_unwrap_val(seed, 42))

    key = _get_cache_key(station_id, days_val, contam_val, seed_val)
    if key in _CACHE:
        return _CACHE[key]

    station_clean = str(station_id).split()[0].upper()
    df_raw = generate_synthetic_weather_data(
        days=days_val,
        interval_minutes=5,
        station_id=station_clean,
        seed=seed_val
    )
    df_injected = inject_anomalies(
        df_raw,
        n_spikes=8,
        n_frozen=8,
        n_drift=8,
        n_gaps=8,
        seed=seed_val
    )

    detector = WeatherAnomalyDetector(
        contamination=contam_val,
        random_state=seed_val
    )
    detector.fit(df_injected)
    df_processed = detector.detect_and_diagnose(df_injected)

    explainer = AnomalyExplainer(detector, background_data=df_injected, sample_size=30)

    entry = {
        "df": df_processed,
        "detector": detector,
        "explainer": explainer,
        "station_id": station_clean,
        "days": days,
        "contamination": contamination,
        "seed": seed,
    }
    _CACHE[key] = entry
    return entry


@router.get("/stations")
def list_deep_dive_stations():
    """List available AWS stations matching weather_anomaly_detection schema."""
    return [
        {
            "id": "AWS-1",
            "name": "AWS-1 (North Field Station)",
            "elevation": "310 m",
            "climate": "Inland Continental Semi-Arid",
            "description": "Baseline regional weather station with wide diurnal temperature oscillation.",
            "is_default": True
        },
        {
            "id": "AWS-2",
            "name": "AWS-2 (Coastal Station)",
            "elevation": "12 m",
            "climate": "Maritime Tropical Humid",
            "description": "Coastal atmospheric station experiencing high relative humidity and salt-fog exposure.",
            "is_default": False
        },
        {
            "id": "AWS-3",
            "name": "AWS-3 (Highland Station)",
            "elevation": "2,450 m",
            "climate": "Alpine / Montane Sub-Zero",
            "description": "High-altitude station with barometric variability and diurnal freeze cycles.",
            "is_default": False
        }
    ]


@router.get("/plotly-chart")
def get_plotly_timeseries_chart(
    station_id: str = "AWS-1",
    days: int = 30,
    contamination: float = 0.055,
    seed: int = 42,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Returns the EXACT 4-row synchronized Plotly subplots figure from weather_anomaly_detection/dashboard.py.
    """
    days_val = int(_unwrap_val(days, 30))
    contam_val = float(_unwrap_val(contamination, 0.055))
    seed_val = int(_unwrap_val(seed, 42))

    data = _get_or_create_processed_data(station_id, days_val, contam_val, seed_val)
    df_data = data["df"]

    min_date = df_data["timestamp"].min().date()
    max_date = df_data["timestamp"].max().date()

    start_date_val = _unwrap_val(start_date, None)
    end_date_val = _unwrap_val(end_date, None)

    if start_date_val:
        start_date_sel = pd.Timestamp(start_date_val).date()
    else:
        start_date_sel = min_date

    if end_date_val:
        end_date_sel = pd.Timestamp(end_date_val).date()
    else:
        end_date_sel = min_date + pd.Timedelta(days=7)

    if start_date_sel > end_date_sel:
        start_date_sel, end_date_sel = end_date_sel, start_date_sel

    mask_time = (df_data["timestamp"].dt.date >= start_date_sel) & (df_data["timestamp"].dt.date <= end_date_sel)
    df_sub = df_data[mask_time].copy().reset_index(drop=True)

    if len(df_sub) == 0:
        df_sub = df_data.head(2016).copy().reset_index(drop=True)

    # 4-Row Synchronized Plotly Subplots (Exact match to weather_anomaly_detection/dashboard.py)
    fig_timeseries = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "<b>1. Temperature Telemetry (°C)</b>",
            "<b>2. Barometric Pressure Telemetry (hPa)</b>",
            "<b>3. Relative Humidity Telemetry (%)</b>",
            "<b>4. Rolling Sensor Health Index (%)</b>"
        )
    )
    # Anomalies subset matching weather_anomaly_detection/dashboard.py
    anom_sub = df_sub[df_sub["pred_is_anomaly"] == 1]

    # --- 1. Temperature Subplot ---
    fig_timeseries.add_trace(
        go.Scatter(
            x=df_sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            y=df_sub["temperature"],
            mode="lines",
            name="Temperature (°C)",
            line=dict(color="#F97316", width=1.8),
            hovertemplate="Time: %{x}<br>Temp: %{y:.2f} °C<extra></extra>"
        ),
        row=1, col=1
    )
    if len(anom_sub) > 0:
        fig_timeseries.add_trace(
            go.Scatter(
                x=anom_sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                y=anom_sub["temperature"],
                mode="markers",
                name="Flagged Anomaly",
                marker=dict(color="#DC2626", size=7, symbol="circle", line=dict(color="#7F1D1D", width=1)),
                hovertemplate="<b>ANOMALY DETECTED</b><br>Time: %{x}<br>Temp: %{y:.2f} °C<br>Diagnosis: %{text}<extra></extra>",
                text=anom_sub["pred_anomaly_type"]
            ),
            row=1, col=1
        )

    # --- 2. Pressure Subplot (Exact match to weather_anomaly_detection/dashboard.py) ---
    fig_timeseries.add_trace(
        go.Scatter(
            x=df_sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            y=df_sub["pressure"],
            mode="lines",
            name="Pressure (hPa)",
            line=dict(color="#2563EB", width=1.8),
            hovertemplate="Time: %{x}<br>Pressure: %{y:.2f} hPa<extra></extra>"
        ),
        row=2, col=1
    )
    if len(anom_sub) > 0:
        fig_timeseries.add_trace(
            go.Scatter(
                x=anom_sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                y=anom_sub["pressure"],
                mode="markers",
                name="Flagged Anomaly",
                showlegend=False,
                marker=dict(color="#DC2626", size=7, symbol="circle", line=dict(color="#7F1D1D", width=1)),
                hovertemplate="<b>ANOMALY DETECTED</b><br>Time: %{x}<br>Pressure: %{y:.2f} hPa<extra></extra>"
            ),
            row=2, col=1
        )

    # --- 3. Humidity Subplot (Original weather_anomaly_detection logic) ---
    fig_timeseries.add_trace(
        go.Scatter(
            x=df_sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            y=df_sub["humidity"],
            mode="lines",
            name="Humidity (%)",
            line=dict(color="#06B6D4", width=1.8),
            hovertemplate="Time: %{x}<br>Humidity: %{y:.2f} %<extra></extra>"
        ),
        row=3, col=1
    )
    if len(anom_sub) > 0:
        fig_timeseries.add_trace(
            go.Scatter(
                x=anom_sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                y=anom_sub["humidity"],
                mode="markers",
                name="Flagged Anomaly",
                showlegend=False,
                marker=dict(color="#DC2626", size=7, symbol="circle", line=dict(color="#7F1D1D", width=1)),
                hovertemplate="<b>ANOMALY DETECTED</b><br>Time: %{x}<br>Humidity: %{y:.2f} %<extra></extra>"
            ),
            row=3, col=1
        )

    # --- 4. Sensor Health Score Subplot ---
    fig_timeseries.add_trace(
        go.Scatter(
            x=df_sub["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            y=df_sub["sensor_health"],
            mode="lines",
            name="Sensor Health (%)",
            line=dict(color="#10B981", width=2.0),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.15)",
            hovertemplate="Time: %{x}<br>Health Score: %{y:.1f} %<extra></extra>"
        ),
        row=4, col=1
    )

    fig_timeseries.update_yaxes(title_text="Temp (°C)", row=1, col=1)
    fig_timeseries.update_yaxes(title_text="Pressure (hPa)", row=2, col=1)
    fig_timeseries.update_yaxes(title_text="Humidity (%)", range=[0, 105], row=3, col=1)
    fig_timeseries.update_yaxes(title_text="Health (%)", range=[0, 105], row=4, col=1)

    fig_timeseries.update_layout(
        height=720,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    import json
    fig_json = json.loads(fig_timeseries.to_json())
    return {
        "data": fig_json["data"],
        "layout": fig_json["layout"],
        "min_date": min_date.strftime("%Y-%m-%d"),
        "max_date": max_date.strftime("%Y-%m-%d"),
        "start_date": start_date_sel.strftime("%Y-%m-%d"),
        "end_date": end_date_sel.strftime("%Y-%m-%d")
    }


@router.get("/telemetry")
def get_telemetry_deep_dive(
    station_id: str = "AWS-1",
    days: int = 30,
    contamination: float = 0.055,
    seed: int = 42,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Returns full multivariate telemetry time-series, synchronized 4-channel metrics,
    KPI summary, and red anomaly flags.
    """
    days_val = int(_unwrap_val(days, 30))
    contam_val = float(_unwrap_val(contamination, 0.055))
    seed_val = int(_unwrap_val(seed, 42))

    data = _get_or_create_processed_data(station_id, days_val, contam_val, seed_val)
    df = data["df"].copy()

    total_readings = len(df)
    total_anomalies = int(df["pred_is_anomaly"].sum())
    anomaly_rate = round((total_anomalies / total_readings) * 100.0, 2) if total_readings > 0 else 0.0
    latest_health = float(df["sensor_health"].iloc[-1]) if total_readings > 0 else 100.0
    avg_health = round(float(df["sensor_health"].mean()), 1) if total_readings > 0 else 100.0

    if latest_health >= 85.0:
        health_status = "OPTIMAL"
    elif latest_health >= 60.0:
        health_status = "DEGRADED"
    else:
        health_status = "CRITICAL"

    # Failure Mode Distribution
    anom_counts = df[df["pred_is_anomaly"] == 1]["pred_anomaly_type"].value_counts().to_dict()
    failure_distribution = [
        {
            "name": k,
            "count": int(v),
            "pct": f"{(v / total_anomalies * 100.0):.1f}%" if total_anomalies > 0 else "0.0%"
        }
        for k, v in anom_counts.items()
    ]

    # Filter date range if provided
    start_date_val = _unwrap_val(start_date, None)
    end_date_val = _unwrap_val(end_date, None)
    if start_date_val:
        df = df[df["timestamp"] >= pd.Timestamp(start_date_val)]
    if end_date_val:
        df = df[df["timestamp"] <= pd.Timestamp(end_date_val) + pd.Timedelta(days=1)]

    # Time series points
    time_series = []
    anomalies_list = []

    for idx, row in df.iterrows():
        ts_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
        is_anom = int(row["pred_is_anomaly"]) == 1
        anom_type = str(row["pred_anomaly_type"])
        pred_feat = str(row.get("pred_anomaly_feature", "none")).lower()
        gt_feat = str(row.get("anomaly_feature", "none")).lower()

        pt = {
            "index": int(idx),
            "timestamp": ts_str,
            "temperature": None if pd.isna(row["temperature"]) else round(float(row["temperature"]), 2),
            "pressure": None if pd.isna(row["pressure"]) else round(float(row["pressure"]), 2),
            "humidity": None if pd.isna(row["humidity"]) else round(float(row["humidity"]), 2),
            "sensor_health": round(float(row["sensor_health"]), 1),
            "is_anomaly": is_anom,
            "is_temp_anomaly": is_anom,
            "is_pressure_anomaly": is_anom,
            "is_humidity_anomaly": is_anom,
            "affected_feature": pred_feat if pred_feat != "none" else gt_feat,
            "anomaly_type": anom_type,
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "ground_truth": int(row.get("is_anomaly", 0)) == 1,
            "ground_truth_type": str(row.get("anomaly_type", "Normal")),
        }
        time_series.append(pt)

        if is_anom:
            anomalies_list.append({
                "index": int(idx),
                "timestamp": ts_str,
                "station_id": str(row["station_id"]),
                "temperature": pt["temperature"],
                "pressure": pt["pressure"],
                "humidity": pt["humidity"],
                "pred_anomaly_type": anom_type,
                "affected_feature": pt["affected_feature"],
                "is_temp_anomaly": is_anom,
                "is_pressure_anomaly": is_anom,
                "is_humidity_anomaly": is_anom,
                "confidence": pt["confidence"],
                "anomaly_score": pt["anomaly_score"],
                "sensor_health": pt["sensor_health"],
                "ground_truth": pt["ground_truth"],
                "ground_truth_type": pt["ground_truth_type"],
            })

    min_date = df["timestamp"].min().strftime("%Y-%m-%d") if len(df) > 0 else None
    max_date = df["timestamp"].max().strftime("%Y-%m-%d") if len(df) > 0 else None

    return {
        "station_id": data["station_id"],
        "days": days_val,
        "contamination": contam_val,
        "seed": seed_val,
        "min_date": min_date,
        "max_date": max_date,
        "kpi": {
            "total_readings": total_readings,
            "total_anomalies": total_anomalies,
            "anomaly_rate": anomaly_rate,
            "latest_health": latest_health,
            "avg_health": avg_health,
            "health_status": health_status,
        },
        "failure_distribution": failure_distribution,
        "time_series": time_series,
        "anomalies": anomalies_list,
    }


@router.get("/flagged-log")
def get_flagged_anomalies_log(
    station_id: str = "AWS-1",
    days: int = 30,
    contamination: float = 0.055,
    seed: int = 42,
    anomaly_types: Optional[str] = None,
    min_confidence: float = 0.0,
    only_anomalies: bool = True,
    export_csv: bool = False
):
    """
    Returns filtered Flagged Telemetry Anomalies Log with CSV export support.
    """
    days_val = int(_unwrap_val(days, 30))
    contam_val = float(_unwrap_val(contamination, 0.055))
    seed_val = int(_unwrap_val(seed, 42))
    anomaly_types_val = _unwrap_val(anomaly_types, None)
    min_conf_val = float(_unwrap_val(min_confidence, 0.0))
    only_anom_val = bool(_unwrap_val(only_anomalies, True))
    export_csv_val = bool(_unwrap_val(export_csv, False))

    data = _get_or_create_processed_data(station_id, days_val, contam_val, seed_val)
    df = data["df"].copy()

    if only_anom_val:
        df = df[df["pred_is_anomaly"] == 1]

    if anomaly_types_val and isinstance(anomaly_types_val, str):
        selected_types = [t.strip() for t in anomaly_types_val.split(",") if t.strip()]
        if selected_types:
            df = df[df["pred_anomaly_type"].isin(selected_types)]

    if min_conf_val > 0:
        df = df[df["confidence"] >= min_conf_val]

    # Display columns
    display_cols = [
        "timestamp", "station_id", "temperature", "pressure", "humidity",
        "pred_anomaly_type", "confidence", "anomaly_score", "sensor_health"
    ]
    if "anomaly_type" in df.columns:
        display_cols.insert(6, "anomaly_type")

    # CSV Export Mode
    if export_csv_val:
        rename_dict = {
            "timestamp": "Timestamp",
            "station_id": "Station",
            "temperature": "Temp (°C)",
            "pressure": "Pressure (hPa)",
            "humidity": "Humidity (%)",
            "pred_anomaly_type": "Predicted Diagnosis",
            "anomaly_type": "Ground Truth",
            "confidence": "Confidence",
            "anomaly_score": "Anomaly Score",
            "sensor_health": "Health (%)"
        }
        df_export = df[display_cols].rename(columns=rename_dict)
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        return PlainTextResponse(
            csv_buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=weather_anomalies_{data['station_id']}.csv"}
        )

    # JSON response
    rows = []
    for idx, row in df.iterrows():
        rows.append({
            "index": int(idx),
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "station_id": str(row["station_id"]),
            "temperature": None if pd.isna(row["temperature"]) else round(float(row["temperature"]), 2),
            "pressure": None if pd.isna(row["pressure"]) else round(float(row["pressure"]), 2),
            "humidity": None if pd.isna(row["humidity"]) else round(float(row["humidity"]), 2),
            "pred_anomaly_type": str(row["pred_anomaly_type"]),
            "ground_truth": str(row.get("anomaly_type", "Normal")),
            "confidence": round(float(row["confidence"]), 3),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "sensor_health": round(float(row["sensor_health"]), 1),
            "is_anomaly": int(row["pred_is_anomaly"]) == 1,
        })

    available_types = [str(t) for t in data["df"]["pred_anomaly_type"].unique().tolist()]

    return {
        "station_id": data["station_id"],
        "total_filtered": len(rows),
        "available_types": available_types,
        "rows": rows
    }


@router.get("/explain")
def explain_anomaly_point(
    station_id: str = "AWS-1",
    days: int = 30,
    contamination: float = 0.055,
    seed: int = 42,
    index: Optional[int] = None,
):
    """
    Compute real 9-feature SHAP TreeExplainer feature attributions using weather_anomaly_detection explainer.
    """
    days_val = int(_unwrap_val(days, 30))
    contam_val = float(_unwrap_val(contamination, 0.055))
    seed_val = int(_unwrap_val(seed, 42))
    index_val = _unwrap_val(index, None)
    if index_val is not None:
        index_val = int(index_val)

    data = _get_or_create_processed_data(station_id, days_val, contam_val, seed_val)
    df = data["df"]
    explainer = data["explainer"]

    if len(df) == 0:
        raise HTTPException(404, "No telemetry data available.")

    if index_val is None or index_val not in df.index:
        anom_rows = df[df["pred_is_anomaly"] == 1]
        if len(anom_rows) > 0:
            target_idx = int(anom_rows.index[0])
        else:
            target_idx = int(df.index[0])
    else:
        target_idx = index_val

    target_row = df.loc[target_idx]
    pred_type = str(target_row["pred_anomaly_type"])

    shap_dict = explainer.explain_point(target_row)

    # Human-readable feature labels
    feature_labels = {
        "temperature": "Temperature (°C)",
        "pressure": "Pressure (hPa)",
        "humidity": "Humidity (%)",
        "temperature_rolling_mean_20": "Temp Moving Avg (100 min)",
        "temperature_deviation_20": "Temp Local Deviation",
        "pressure_rolling_mean_20": "Pressure Moving Avg (100 min)",
        "pressure_deviation_20": "Pressure Local Deviation",
        "humidity_rolling_mean_20": "Humidity Moving Avg (100 min)",
        "humidity_deviation_20": "Humidity Local Deviation",
    }

    features_list = []
    top_feature = None
    top_feature_label = None
    max_abs = -1.0

    for feat, impact in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True):
        abs_mag = round(abs(float(impact)), 4)
        if abs_mag > max_abs:
            max_abs = abs_mag
            top_feature = feat
            top_feature_label = feature_labels.get(feat, feat)

        color_type = "primary" if feat in ["temperature", "pressure", "humidity"] and abs_mag >= 0.25 else (
            "deviation" if "deviation" in feat else "baseline"
        )

        features_list.append({
            "feature": feat,
            "label": feature_labels.get(feat, feat),
            "shap_value": round(float(impact), 4),
            "abs_magnitude": abs_mag,
            "color_type": color_type
        })

    if features_list:
        features_list[0]["color_type"] = "primary"

    action_map = {
        "Spike": "Transient electrical impulse or EMI disturbance. Filter outlier and inspect power supply grounding.",
        "Frozen": "Sensor communication bus locked or transducer mechanically stuck. Perform remote reboot or check I2C/RS485 interface.",
        "Drift": "Sensor calibration offset degradation or psychrometric wick contamination. Recalibrate sensor against reference standard.",
        "Communication gap": "Telemetry packet loss. Check 4G/LoRaWAN signal strength, solar battery voltage, and gateway status.",
        "Multivariate Outlier": "Thermodynamic inconsistency detected between Temperature and Humidity correlation. Inspect ventilation fan."
    }
    rec_action = action_map.get(pred_type, "Inspect sensor telemetry and verify against neighboring weather stations.")

    event_payload = {
        "index": target_idx,
        "timestamp": target_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "station_id": str(target_row["station_id"]),
        "temperature": None if pd.isna(target_row["temperature"]) else round(float(target_row["temperature"]), 2),
        "pressure": None if pd.isna(target_row["pressure"]) else round(float(target_row["pressure"]), 2),
        "humidity": None if pd.isna(target_row["humidity"]) else round(float(target_row["humidity"]), 2),
        "pred_anomaly_type": pred_type,
        "ground_truth": str(target_row.get("anomaly_type", "Normal")),
        "confidence": round(float(target_row["confidence"]), 4),
        "anomaly_score": round(float(target_row["anomaly_score"]), 4),
        "sensor_health": round(float(target_row["sensor_health"]), 1),
        "is_anomaly": bool(int(target_row.get("pred_is_anomaly", 0)) == 1),
    }

    return {
        "event": event_payload,
        "explanation": {
            "features": features_list,
            "top_contributor": top_feature,
            "top_contributor_label": top_feature_label,
            "recommended_action": rec_action,
            "predicted_anomaly_type": pred_type,
        }
    }


@router.post("/re-simulate")
def re_simulate_data(
    station_id: str = "AWS-1",
    days: int = 30,
    contamination: float = 0.055,
    seed: int = 42
):
    """
    Clear cache and trigger a fresh simulation run.
    """
    key = _get_cache_key(station_id, days, contamination, seed)
    if key in _CACHE:
        del _CACHE[key]
    data = _get_or_create_processed_data(station_id, days, contamination, seed)
    return {
        "status": "success",
        "station_id": data["station_id"],
        "total_readings": len(data["df"]),
        "detected_anomalies": int(data["df"]["pred_is_anomaly"].sum())
    }
