import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from tensorflow import keras

st.set_page_config(page_title="Materials Property Predictor", layout="wide")

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(".")
PRESSURE_COL_NAME = "applied_pressure_GPa"

ALL_ELEMENTS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

COMPOSITION_FEATURES = [f"wtpct_{el}" for el in ALL_ELEMENTS]

MODEL_SPECS = {
    "bulk_modulus_GPa": {
        "label": "Bulk Modulus (GPa)",
        "model_file": "bulk_modulus_dl_model.keras",
        "preprocessor_file": "bulk_modulus_dl_preprocessor.pkl",
        "prediction_col": "predicted_bulk_modulus_GPa",
        "log_target": False,
    },
    "shear_modulus_GPa": {
        "label": "Shear Modulus (GPa)",
        "model_file": "shear_modulus_dl_model.keras",
        "preprocessor_file": "shear_modulus_dl_preprocessor.pkl",
        "prediction_col": "predicted_shear_modulus_GPa",
        "log_target": False,
    },
    "youngs_modulus_GPa": {
        "label": "Young's Modulus (GPa)",
        "model_file": "youngs_modulus_dl_model.keras",
        "preprocessor_file": "youngs_modulus_dl_preprocessor.pkl",
        "prediction_col": "predicted_youngs_modulus_GPa",
        "log_target": False,
    },
    "formation_energy_eV_per_atom": {
        "label": "Formation Energy (eV/atom)",
        "model_file": "formation_energy_eV_per_atom_dl_model.keras",
        "preprocessor_file": "formation_energy_eV_per_atom_dl_preprocessor.pkl",
        "prediction_col": "predicted_formation_energy_eV_per_atom",
        "log_target": False,
    },
    "energy_above_hull_eV_per_atom": {
        "label": "Energy Above Hull (eV/atom)",
        "model_file": "energy_above_hull_eV_per_atom_dl_model.keras",
        "preprocessor_file": "energy_above_hull_eV_per_atom_dl_preprocessor.pkl",
        "prediction_col": "predicted_energy_above_hull_eV_per_atom",
        "log_target": True,
    },
}

DEFAULT_DENSITY = 7.85
DEFAULT_PRESSURE = 1.0

# =========================================================
# HELPERS
# =========================================================
@st.cache_resource
def load_model_and_preprocessor(model_path: str, preprocessor_path: str):
    model_file = BASE_DIR / model_path
    preprocessor_file = BASE_DIR / preprocessor_path

    if not model_file.exists() or not preprocessor_file.exists():
        return None, None

    model = keras.models.load_model(model_file)
    preprocessor = joblib.load(preprocessor_file)
    return model, preprocessor


def infer_features():
    numeric_features = ["density"] + COMPOSITION_FEATURES + ["n_elements_present"]
    categorical_features = ["base_metal"]
    feature_cols = numeric_features + categorical_features
    return numeric_features, categorical_features, feature_cols


def align_input_df(input_df: pd.DataFrame, feature_cols):
    df = input_df.copy()
    for col in feature_cols:
        if col not in df.columns:
            if col.startswith("wtpct_"):
                df[col] = 0.0
            else:
                df[col] = np.nan
    return df[feature_cols]


def run_prediction(input_df: pd.DataFrame, model, preprocessor, log_target: bool):
    X_processed = preprocessor.transform(input_df)
    preds = model.predict(X_processed, verbose=0).ravel()

    if log_target:
        preds = np.expm1(preds)

    return np.clip(preds, 0, None)


def get_available_models():
    available = {}
    missing = {}

    for target, spec in MODEL_SPECS.items():
        model_exists = (BASE_DIR / spec["model_file"]).exists()
        preprocessor_exists = (BASE_DIR / spec["preprocessor_file"]).exists()

        if model_exists and preprocessor_exists:
            available[target] = spec
        else:
            missing[target] = spec

    return available, missing


def add_factor_of_safety(result_df: pd.DataFrame):
    if "predicted_bulk_modulus_GPa" not in result_df.columns:
        return result_df

    if PRESSURE_COL_NAME not in result_df.columns:
        return result_df

    pressure = pd.to_numeric(result_df[PRESSURE_COL_NAME], errors="coerce")
    bulk = pd.to_numeric(result_df["predicted_bulk_modulus_GPa"], errors="coerce")

    fos = np.where(pressure > 0, bulk / pressure, np.nan)
    result_df["factor_of_safety"] = fos
    return result_df


def build_sample_template(feature_cols, categorical_features, default_pressure):
    template = {}
    for col in feature_cols:
        if col.startswith("wtpct_"):
            template[col] = 0.0
        elif col == "density":
            template[col] = DEFAULT_DENSITY
        elif col == "n_elements_present":
            template[col] = 1
        elif col in categorical_features:
            template[col] = ""
        else:
            template[col] = 0.0

    template[PRESSURE_COL_NAME] = default_pressure
    return pd.DataFrame([template])


def infer_base_metal_from_composition(composition_dict):
    nonzero_items = {k: float(v) for k, v in composition_dict.items() if float(v) > 0}
    if not nonzero_items:
        return np.nan

    highest_col = max(nonzero_items, key=nonzero_items.get)
    return highest_col.replace("wtpct_", "")


def count_elements_from_composition(composition_dict):
    return int(sum(float(v) > 0 for v in composition_dict.values()))


def auto_assign_base_metal_for_dataframe(df, composition_features):
    out = df.copy()

    def get_base(row):
        row_values = row[composition_features].fillna(0)
        if (row_values > 0).sum() == 0:
            return np.nan
        highest_col = row_values.idxmax()
        return highest_col.replace("wtpct_", "")

    out["base_metal"] = out.apply(get_base, axis=1)
    return out


def auto_assign_n_elements_for_dataframe(df, composition_features):
    out = df.copy()
    out["n_elements_present"] = (out[composition_features].fillna(0) > 0).sum(axis=1)
    return out


def validate_single_input_row(row_dict, composition_features):
    messages = []
    errors = []

    total_wt = sum(float(row_dict.get(col, 0.0)) for col in composition_features)

    if 99 <= total_wt <= 100:
        status_type = "success"
        status_text = f"Composition total = {total_wt:.2f}%"
    else:
        status_type = "error"
        status_text = f"Composition total is {total_wt:.2f}%. It must be between 99% and 100%."

    density_value = pd.to_numeric(pd.Series([row_dict.get("density", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(density_value) or density_value <= 0:
        errors.append("Density should be greater than 0.")

    pressure_value = pd.to_numeric(pd.Series([row_dict.get(PRESSURE_COL_NAME, np.nan)]), errors="coerce").iloc[0]
    if pd.isna(pressure_value) or pressure_value <= 0:
        messages.append("Applied pressure is 0 or invalid, so Factor of Safety will not be computed.")

    selected_nonzero = [col for col in composition_features if float(row_dict.get(col, 0.0)) > 0]
    if len(selected_nonzero) == 0:
        errors.append("Please enter at least one element percentage.")

    n_elements_value = pd.to_numeric(pd.Series([row_dict.get("n_elements_present", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(n_elements_value) or int(n_elements_value) < 1:
        errors.append("At least one element must be present.")

    return total_wt, status_type, status_text, messages, errors


def validate_uploaded_csv(df, feature_cols, composition_features, default_pressure):
    report = {}

    missing_feature_cols = [col for col in feature_cols if col not in df.columns]
    added_cols = []

    out = df.copy()

    for col in missing_feature_cols:
        if col.startswith("wtpct_"):
            out[col] = 0.0
        elif col == "density":
            out[col] = np.nan
        elif col == "n_elements_present":
            out[col] = np.nan
        elif col == "base_metal":
            out[col] = np.nan
        else:
            out[col] = np.nan
        added_cols.append(col)

    if PRESSURE_COL_NAME not in out.columns:
        out[PRESSURE_COL_NAME] = default_pressure
        added_cols.append(PRESSURE_COL_NAME)

    out = auto_assign_base_metal_for_dataframe(out, composition_features)
    out = auto_assign_n_elements_for_dataframe(out, composition_features)

    composition_total = out[composition_features].fillna(0).sum(axis=1)
    valid_composition = composition_total.between(99, 100)

    valid_density = pd.to_numeric(out.get("density", np.nan), errors="coerce").fillna(-1) > 0
    valid_rows_mask = valid_density & valid_composition

    report["rows_uploaded"] = len(out)
    report["rows_valid"] = int(valid_rows_mask.sum())
    report["rows_invalid"] = int((~valid_rows_mask).sum())
    report["missing_feature_columns"] = missing_feature_cols
    report["added_columns"] = added_cols
    report["rows_composition_valid"] = int(valid_composition.sum())
    report["rows_composition_invalid"] = int((~valid_composition).sum())

    return out, report


# =========================================================
# FEATURES ONLY FROM CODE
# =========================================================
numeric_features, categorical_features, feature_cols = infer_features()
composition_features = COMPOSITION_FEATURES
element_options = ALL_ELEMENTS

available_models, missing_models = get_available_models()

# =========================================================
# LANDING SECTION
# =========================================================
if "input_mode" not in st.session_state:
    st.session_state.input_mode = None

st.markdown(
    """
    <div style="text-align:center; margin-top:40px; margin-bottom:10px;">
        <h1>Materials Property Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align:center; margin-bottom:30px; color:gray;">
        Predict material properties and calculate factor of safety from material composition
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align:center; margin-bottom:20px;">
        <h3>Choose input type</h3>
    </div>
    """,
    unsafe_allow_html=True
)

left_space, col1, col2, right_space = st.columns([1.2, 2, 2, 1.2])

with col1:
    st.markdown(
        """
        <div style="border:1px solid #ddd; border-radius:14px; padding:24px; text-align:center; min-height:160px;">
            <h3>Single Material</h3>
            <p>Enter one material manually and get instant predictions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Single Material", use_container_width=True, key="single_material_home"):
        st.session_state.input_mode = "Single Material"

with col2:
    st.markdown(
        """
        <div style="border:1px solid #ddd; border-radius:14px; padding:24px; text-align:center; min-height:160px;">
            <h3>CSV Upload</h3>
            <p>Upload multiple rows and predict properties in bulk.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("CSV Upload", use_container_width=True, key="csv_upload_home"):
        st.session_state.input_mode = "CSV Upload"

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("App Settings")

    if available_models:
        selected_targets = st.multiselect(
            "Choose prediction targets",
            options=list(available_models.keys()),
            default=list(available_models.keys()),
            format_func=lambda x: available_models[x]["label"],
        )
    else:
        selected_targets = []

    st.markdown("---")
    st.subheader("Factor of Safety")
    st.caption("FOS = predicted bulk modulus / applied pressure")
    default_pressure = st.number_input(
        "Default applied pressure (GPa)",
        min_value=0.0,
        value=DEFAULT_PRESSURE,
        step=0.1
    )

    st.markdown("---")
    st.subheader("Available models")
    if available_models:
        for _, spec in available_models.items():
            st.success(spec["label"])
    else:
        st.warning("No complete model files found.")

    if missing_models:
        st.markdown("---")
        st.subheader("Missing files")
        for _, spec in missing_models.items():
            st.error(spec["label"])

# =========================================================
# SINGLE INPUT MODE
# =========================================================
def single_material_input(
    feature_cols,
    composition_features,
    element_options,
    default_pressure
):
    st.subheader("Single Material Input")

    if "selected_elements" not in st.session_state:
        st.session_state.selected_elements = ["Fe", "Ni"] if "Fe" in element_options and "Ni" in element_options else []
    if "element_values" not in st.session_state:
        st.session_state.element_values = {}

    with st.container():
        col_a, col_b, col_c = st.columns([1.2, 1.6, 1.1])

        with col_a:
            st.markdown("### Basic material information")
            st.caption("Base metal and number of elements are assigned automatically.")

            density = st.number_input("Density", min_value=0.0, value=DEFAULT_DENSITY, step=0.01)

            applied_pressure = st.number_input(
                "Applied pressure (GPa)",
                min_value=0.0,
                value=float(default_pressure),
                step=0.1
            )

        with col_b:
            st.markdown("### Composition input")
            st.caption("Choose only the elements you want, then enter their wt% values.")

            selected_elements = st.multiselect(
                "Select elements",
                options=element_options,
                default=st.session_state.selected_elements
            )
            st.session_state.selected_elements = selected_elements

            wt_values_by_element = {}

            if selected_elements:
                entry_cols = st.columns(2)
                for i, elem in enumerate(selected_elements):
                    key_name = f"wt_input_{elem}"
                    default_value = float(st.session_state.element_values.get(elem, 0.0))
                    with entry_cols[i % 2]:
                        wt_values_by_element[elem] = st.number_input(
                            f"{elem} wt%",
                            min_value=0.0,
                            max_value=100.0,
                            value=default_value,
                            step=0.1,
                            key=key_name
                        )
            else:
                st.info("Select one or more elements to enter composition.")

            for elem in selected_elements:
                wt_values_by_element[elem] = float(st.session_state.get(f"wt_input_{elem}", 0.0))
                st.session_state.element_values[elem] = wt_values_by_element[elem]

        with col_c:
            st.markdown("### Validation summary")

            composition_dict = {col: 0.0 for col in composition_features}
            for elem in selected_elements:
                col_name = f"wtpct_{elem}"
                if col_name in composition_dict:
                    composition_dict[col_name] = float(wt_values_by_element.get(elem, 0.0))

            auto_base_metal = infer_base_metal_from_composition(composition_dict)
            auto_n_elements = count_elements_from_composition(composition_dict)

            row_preview = {col: 0.0 for col in feature_cols}
            row_preview["base_metal"] = auto_base_metal
            row_preview["density"] = density
            row_preview["n_elements_present"] = auto_n_elements
            row_preview.update(composition_dict)
            row_preview[PRESSURE_COL_NAME] = applied_pressure

            st.write(f"**Detected base metal:** {auto_base_metal}")
            st.write(f"**Detected number of elements:** {auto_n_elements}")

            total_wt, status_type, status_text, messages, errors = validate_single_input_row(
                row_preview, composition_features
            )

            st.metric("Total wt%", f"{total_wt:.2f}")

            if status_type == "success":
                st.success(status_text)
            else:
                st.error(status_text)

            if messages:
                for msg in messages:
                    st.info(msg)

            if errors:
                for err in errors:
                    st.error(err)
            else:
                st.success("Input looks ready for prediction.")

    if st.button("Predict Properties", type="primary"):
        if errors or status_type != "success":
            st.error("Please make sure the composition total is between 99% and 100% and fix all input issues.")
            return None

        final_row = {col: 0.0 for col in feature_cols}
        final_row["base_metal"] = infer_base_metal_from_composition(composition_dict)
        final_row["density"] = density
        final_row["n_elements_present"] = count_elements_from_composition(composition_dict)
        final_row.update(composition_dict)
        final_row[PRESSURE_COL_NAME] = applied_pressure

        return pd.DataFrame([final_row])

    return None

# =========================================================
# CSV MODE
# =========================================================
def csv_input_mode(default_pressure, feature_cols, composition_features, categorical_features):
    st.subheader("CSV Upload")

    sample_template = build_sample_template(feature_cols, categorical_features, default_pressure)
    template_bytes = sample_template.to_csv(index=False).encode("utf-8")

    top_left, top_right = st.columns([1.2, 1.8])

    with top_left:
        st.download_button(
            label="Download sample CSV template",
            data=template_bytes,
            file_name="materials_input_template.csv",
            mime="text/csv"
        )

    with top_right:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is None:
        return None

    raw_df = pd.read_csv(uploaded_file)
    validated_df, report = validate_uploaded_csv(
        raw_df,
        feature_cols=feature_cols,
        composition_features=composition_features,
        default_pressure=default_pressure
    )

    st.markdown("### Upload status")
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Rows uploaded", report["rows_uploaded"])
    stat2.metric("Rows valid", report["rows_valid"])
    stat3.metric("Rows invalid", report["rows_invalid"])
    stat4.metric("Rows with valid composition", report["rows_composition_valid"])

    if report["missing_feature_columns"]:
        st.warning(
            "Missing feature columns were added automatically: "
            + ", ".join(report["missing_feature_columns"])
        )

    if PRESSURE_COL_NAME in report["added_columns"]:
        st.info(f"'{PRESSURE_COL_NAME}' was missing, so the default applied pressure was added.")

    if report["rows_composition_invalid"] > 0:
        st.error(
            f"{report['rows_composition_invalid']} rows have composition totals outside the allowed 99%–100% range."
        )

    st.markdown("### Preview")
    st.dataframe(validated_df.head())

    if report["rows_valid"] == 0:
        st.error("No valid rows found. Please correct the CSV before prediction.")
        return None

    return validated_df

# =========================================================
# MAIN INPUT FLOW
# =========================================================
if not available_models:
    st.warning("Put your five .keras and .pkl files in the same folder as this app, then rerun Streamlit.")
    st.stop()

if not selected_targets:
    st.warning("Choose at least one available target from the sidebar.")
    st.stop()

input_mode = st.session_state.input_mode

if input_mode == "Single Material":
    input_df = single_material_input(
        feature_cols=feature_cols,
        composition_features=composition_features,
        element_options=element_options,
        default_pressure=default_pressure
    )
elif input_mode == "CSV Upload":
    input_df = csv_input_mode(
        default_pressure=default_pressure,
        feature_cols=feature_cols,
        composition_features=composition_features,
        categorical_features=categorical_features
    )
else:
    input_df = None

# =========================================================
# PREDICTIONS
# =========================================================
if input_df is not None:
    aligned_df = align_input_df(input_df, feature_cols)
    result_df = input_df.copy()

    with st.spinner("Running models..."):
        for target in selected_targets:
            spec = available_models[target]
            model, preprocessor = load_model_and_preprocessor(
                spec["model_file"],
                spec["preprocessor_file"]
            )

            if model is None or preprocessor is None:
                st.warning(f"Skipping {spec['label']} because files could not be loaded.")
                continue

            try:
                preds = run_prediction(aligned_df, model, preprocessor, spec["log_target"])
                result_df[spec["prediction_col"]] = preds
            except Exception as e:
                st.error(f"Prediction failed for {spec['label']}: {e}")

        result_df = add_factor_of_safety(result_df)

    st.subheader("Prediction Results")
    st.dataframe(result_df)

    if len(result_df) == 1:
        st.subheader("Predicted Values")

        metric_items = []

        if "predicted_bulk_modulus_GPa" in result_df.columns:
            metric_items.append(("Bulk Modulus (GPa)", result_df.iloc[0]["predicted_bulk_modulus_GPa"]))
        if "predicted_shear_modulus_GPa" in result_df.columns:
            metric_items.append(("Shear Modulus (GPa)", result_df.iloc[0]["predicted_shear_modulus_GPa"]))
        if "predicted_youngs_modulus_GPa" in result_df.columns:
            metric_items.append(("Young's Modulus (GPa)", result_df.iloc[0]["predicted_youngs_modulus_GPa"]))
        if "predicted_formation_energy_eV_per_atom" in result_df.columns:
            metric_items.append(("Formation Energy", result_df.iloc[0]["predicted_formation_energy_eV_per_atom"]))
        if "predicted_energy_above_hull_eV_per_atom" in result_df.columns:
            metric_items.append(("Energy Above Hull", result_df.iloc[0]["predicted_energy_above_hull_eV_per_atom"]))
        if "factor_of_safety" in result_df.columns:
            metric_items.append(("Factor of Safety", result_df.iloc[0]["factor_of_safety"]))

        cols = st.columns(3)
        for i, (label, value) in enumerate(metric_items):
            with cols[i % 3]:
                if pd.notna(value):
                    st.metric(label, f"{value:.4f}")
                else:
                    st.metric(label, "N/A")

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv_bytes,
        file_name="materials_predictions.csv",
        mime="text/csv",
    )
