"""Streamlit interface for the existing SoilDoctor Flask/AI backend."""

import base64
import json

import streamlit as st

from app import app as flask_app


st.set_page_config(page_title="SoilDoctor", page_icon="🌱", layout="wide")

CROPS = ["Tomato", "Rice", "Chilli", "Sugarcane", "Blackgram", "Soybean"]
NUTRIENT_LEVELS = ["Normal", "Low", "High"]


def api_get(path):
    """Call the Flask API in-process so its scan and database logic is reused."""
    with flask_app.test_client() as client:
        response = client.get(path)
        data = response.get_json(silent=True) or {}
        if response.status_code >= 400 or not data.get("success", True):
            raise RuntimeError(data.get("error", f"Request failed ({response.status_code})"))
        return data


def api_post(path, payload):
    with flask_app.test_client() as client:
        response = client.post(path, json=payload)
        data = response.get_json(silent=True) or {}
        if response.status_code >= 400 or not data.get("success"):
            raise RuntimeError(data.get("error", f"Request failed ({response.status_code})"))
        return data["data"]


def load_fields():
    try:
        return api_get("/api/farms").get("fields", [])
    except Exception as exc:
        st.error(f"Could not load farm fields: {exc}")
        return []


def field_picker(fields, key):
    if not fields:
        st.warning("No saved fields are available. Add a field in the Flask app first.")
        return None
    selected = st.selectbox(
        "Scan area (required)",
        fields,
        format_func=lambda field: (
            f"{field['name']} · {field['area_acres']} acres · {field['crop']}"
        ),
        key=key,
    )
    return selected


def show_image_data_url(value, caption):
    if not value:
        return
    encoded = value.split(",", 1)[-1]
    st.image(base64.b64decode(encoded), caption=caption, use_container_width=True)


def render_plant_result(result):
    st.success(f"Scan #{result.get('scan_id', '—')} completed")
    a, b, c = st.columns(3)
    a.metric("Crop", result.get("crop", "Unknown"))
    b.metric("Diagnosis", result.get("disease_name", "Unknown"))
    c.metric("Confidence", f"{result.get('confidence', 0)}%")
    st.write(f"**Risk:** {result.get('risk_level', 'Unknown')}")
    st.write(f"**Symptoms:** {result.get('symptoms_summary', '—')}")
    st.write(f"**Organic recommendation:** {result.get('organic_solution', '—')}")
    st.write(f"**Other recommendation:** {result.get('mineral_solution', '—')}")
    images = result.get("images", {})
    image_cols = st.columns(3)
    for col, key, caption in zip(
        image_cols,
        ("original", "segmented", "gradcam"),
        ("Uploaded leaf", "Leaf segmentation", "AI attention view"),
    ):
        with col:
            show_image_data_url(images.get(key), caption)
    summary = result.get("explainability", {}).get("ai_summary")
    if summary:
        st.info(summary)


def plant_scan(fields):
    st.subheader("🌿 Crop disease scan")
    st.caption("Choose the crop shown in the photo. SoilDoctor will not auto-detect the crop.")
    with st.form("plant_scan_form"):
        field = field_picker(fields, "plant_field")
        crop = st.selectbox("Crop in the image", CROPS)
        photo = st.file_uploader("Upload a clear leaf photo", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Analyze leaf", type="primary", disabled=not fields)

    if submitted:
        if field is None:
            st.error("Select a saved field before scanning.")
        elif photo is None:
            st.error("Upload a leaf image first.")
        else:
            payload = {
                "image_b64": base64.b64encode(photo.getvalue()).decode("ascii"),
                "crop_type": crop,
                "field_id": field["id"],
                "location_type": "field",
            }
            with st.spinner("Analyzing the leaf…"):
                try:
                    render_plant_result(api_post("/api/scan/plant", payload))
                except Exception as exc:
                    st.error(f"Plant scan failed: {exc}")


def soil_scan(fields):
    st.subheader("🧪 Soil health scan")
    st.caption("The photo estimates visible color and texture. Nutrient and pH values must come from your inputs or sensors.")
    with st.form("soil_scan_form"):
        field = field_picker(fields, "soil_field")
        photo = st.file_uploader("Upload a close-up soil photo", type=["jpg", "jpeg", "png"], key="soil_photo")
        use_sample = st.checkbox("Use the included demo soil photo instead")
        use_readings = st.checkbox("Add simulated sensor readings", value=True)
        lab_data = {}
        if use_readings:
            ph_col, moisture_col = st.columns(2)
            ph_value = ph_col.number_input("pH", min_value=0.0, max_value=14.0, value=6.4, step=0.1)
            moisture = moisture_col.number_input("Moisture (%)", min_value=0, max_value=100, value=28)
            n_col, p_col, k_col = st.columns(3)
            lab_data = {
                "ph_value": ph_value,
                "moisture_pct": moisture,
                "n_status": n_col.selectbox("Nitrogen (N)", NUTRIENT_LEVELS),
                "p_status": p_col.selectbox("Phosphorus (P)", NUTRIENT_LEVELS),
                "k_status": k_col.selectbox("Potassium (K)", NUTRIENT_LEVELS),
                "measurement_source": "Simulated demo sensor readings",
            }
        submitted = st.form_submit_button("Analyze soil", type="primary", disabled=not fields)

    if submitted:
        if field is None:
            st.error("Select a saved field before scanning.")
        elif not use_sample and photo is None:
            st.error("Upload a soil photo or select the demo soil photo.")
        else:
            payload = {
                "field_id": field["id"],
                "location_type": "field",
                "lab_data": lab_data,
            }
            if use_sample:
                payload["sample"] = "loamy_soil"
            else:
                payload["image_b64"] = base64.b64encode(photo.getvalue()).decode("ascii")
            with st.spinner("Analyzing the soil…"):
                try:
                    result = api_post("/api/scan/soil", payload)
                    st.success(f"Scan #{result.get('scan_id', '—')} completed")
                    visual = result.get("visual_analysis", {})
                    nutrients = result.get("nutrient_status", {})
                    left, right = st.columns(2)
                    with left:
                        st.markdown("#### Photo estimate")
                        st.write(f"**Texture:** {visual.get('texture', '—')}")
                        st.write(f"**Color:** {visual.get('color', '—')}")
                        st.write(f"**Surface:** {visual.get('surface', '—')}")
                        st.caption(visual.get("method", ""))
                    with right:
                        st.markdown("#### Supplied readings")
                        st.write(f"**pH:** {nutrients.get('ph_value') or 'Not measured'}")
                        st.write(f"**N / P / K:** {nutrients.get('n_status', '—')} / {nutrients.get('p_status', '—')} / {nutrients.get('k_status', '—')}")
                        st.write(f"**Moisture:** {nutrients.get('moisture_pct') or 'Not measured'}")
                    show_image_data_url(result.get("soil_image"), "Analyzed soil photo")
                    st.markdown("#### Suggestions")
                    for recommendation in result.get("recommendations", []):
                        st.write(f"• {recommendation}")
                except Exception as exc:
                    st.error(f"Soil scan failed: {exc}")


def dashboard():
    st.subheader("Farm overview")
    try:
        data = api_get("/api/dashboard")
        kpi = data.get("kpi", {})
        cols = st.columns(4)
        cols[0].metric("Crop health", f"{kpi['crop_health']}%" if kpi.get("crop_health") is not None else "—")
        cols[1].metric("Soil health", f"{kpi['soil_health']}%" if kpi.get("soil_health") is not None else "—")
        cols[2].metric("Total scans", kpi.get("total_scans", 0))
        cols[3].metric("Active fields", kpi.get("active_fields", 0))
        st.markdown("#### Recent scans")
        rows = data.get("recent_scans", [])
        if rows:
            st.dataframe(
                [{key: row.get(key) for key in ("scan_type", "crop_name", "disease_name", "confidence", "risk_level", "created_at")} for row in rows],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No scans recorded yet.")
    except Exception as exc:
        st.error(f"Could not load dashboard: {exc}")


def fields_and_history(fields):
    st.subheader("Farm fields")
    if fields:
        st.dataframe(
            [{key: field.get(key) for key in ("name", "crop", "area_acres", "status", "last_scan_date")} for field in fields],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No saved fields found.")
    st.subheader("Scan history")
    try:
        rows = api_get("/api/scans").get("scans", [])
        if rows:
            clean_rows = []
            for row in rows:
                clean_rows.append({
                    "Date": row.get("created_at"),
                    "Type": row.get("scan_type"),
                    "Crop / sample": row.get("crop_name"),
                    "Finding": row.get("disease_name"),
                    "Confidence": row.get("confidence"),
                    "Risk": row.get("risk_level"),
                })
            st.dataframe(clean_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No scans recorded yet.")
    except Exception as exc:
        st.error(f"Could not load scan history: {exc}")


st.title("🌱 SoilDoctor")
st.caption("AI-assisted crop and soil health checks")
st.warning("Demo note: soil photo analysis estimates visible texture and color. pH and NPK values are user-entered/simulated unless connected to real sensors.", icon="ℹ️")

fields = load_fields()
page = st.sidebar.radio("Navigate", ["Overview", "Crop scan", "Soil scan", "Fields & history"])

if page == "Overview":
    dashboard()
elif page == "Crop scan":
    plant_scan(fields)
elif page == "Soil scan":
    soil_scan(fields)
else:
    fields_and_history(fields)

st.sidebar.caption("Streamlit interface · Existing Flask API and AI engine")
st.sidebar.caption("Cloud demo data is stored in the app's local SQLite file and may reset on rebuild.")
