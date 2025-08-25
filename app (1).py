# app.py
# Streamlit Financial Fraud Detection Dashboard (working, robust)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx
from io import StringIO
from typing import List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

st.set_page_config(page_title="Financial Fraud Detection", layout="wide")
st.title("💳 Financial Fraud Detection Dashboard")

# ----------------------------- Helpers -----------------------------

def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """Convert typical truthy/falsy strings/numbers to booleans; leave NaN as NaN."""
    if s.dtype == bool:
        return s
    mapping_true = {"true", "1", "yes", "y", "t"}
    mapping_false = {"false", "0", "no", "n", "f"}
    def to_bool(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return bool(int(x))
        xs = str(x).strip().lower()
        if xs in mapping_true:
            return True
        if xs in mapping_false:
            return False
        return np.nan
    return s.map(to_bool)

@st.cache_data(show_spinner=False)
def fetch_api_dataframe(url: str) -> Tuple[pd.DataFrame, str]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    df = pd.DataFrame(data)
    note = f"Fetched {len(df)} rows from API"
    return df, note

@st.cache_data(show_spinner=False)
def read_csv_file(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(StringIO(file_bytes.decode("utf-8", errors="ignore")))

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]
    # Try convert object numerics; leave categorical as is
    for col in df.columns:
        if df[col].dtype == object:
            # Attempt numeric conversion with explicit exception handling
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass  # Keep original data if conversion fails
            else:
                # Try datetime parsing with explicit exception handling
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except (ValueError, TypeError):
                    pass  # Keep original data if datetime parsing fails
    return df

# --------------------------- Sidebar I/O ----------------------------

st.sidebar.header("Data Loading Options")
data_source = st.sidebar.radio("Choose data source:", ("API", "Upload CSV"), key="data_source_selection_1")

_df = None
status_note = ""

if data_source == "API":
    api_url = st.sidebar.text_input("Enter API URL:", "https://sheetdb.io/api/v1/3digbv94ticxe")
    if st.sidebar.button("Fetch Data", use_container_width=True):
        try:
            _df, status_note = fetch_api_dataframe(api_url)
            st.sidebar.success(status_note)
        except Exception as e:
            st.sidebar.error(f"API error: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            _df = read_csv_file(uploaded_file.getvalue())
            st.sidebar.success("CSV file loaded successfully")
        except Exception as e:
            st.sidebar.error(f"CSV read error: {e}")

if _df is not None and len(_df) > 0:
    df = clean_dataframe(_df)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"Shape: {df.shape}")

    # Target handling
    target_col_options = [c for c in df.columns if str(c).lower() in {"is_fraud", "fraud", "label", "target"}]
    default_target = target_col_options[0] if target_col_options else None
    target_col = st.selectbox("Select target column (for supervised):", ["<None>"] + list(df.columns), index=(0 if default_target is None else (list(["<None>"] + list(df.columns)).index(default_target))))

    if target_col != "<None>":
        df[target_col] = _coerce_bool_series(df[target_col]).astype("float").fillna(0.0).astype(int)
        st.info(f"Target '{target_col}' normalized to 0/1.")

    if target_col != "<None>":
        vc = df[target_col].value_counts(dropna=False).sort_index()
        c0 = int(vc.get(0, 0))
        c1 = int(vc.get(1, 0))
        st.write("**Class distribution (0=legit, 1=fraud):**", {0: c0, 1: c1})
        fig = plt.figure()
        plt.bar(["0", "1"], [c0, c1])
        plt.title("Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        st.pyplot(fig, use_container_width=True)

    st.divider()

    # ---------------- Supervised Learning: RandomForest ----------------
    st.subheader("Supervised Model: RandomForest Classifier")
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        rf_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=200, step=50)
    with colB:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    with colC:
        random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1)
    with colD:
        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=10, step=1)

    # Visualization container
    viz_container = st.container()

    if st.button("Train RandomForest Model", use_container_width=True):
        try:
            if target_col == "<None>":
                st.error("Please select a target column above for supervised training.")
            else:
                y = df[target_col]
                X = df.drop(columns=[target_col])
                X = pd.get_dummies(X, drop_first=True)
                X = X.dropna(axis=1, how='all')
                X = X.fillna(X.median(numeric_only=True))
                if X.shape[1] == 0:
                    st.error("No usable features after preprocessing. Check your dataset for valid features.")
                elif len(np.unique(y)) < 2:
                    st.error("Target has only one class; need both 0 and 1 for training.")
                else:
                    # Feature selection using importance
                    base_model = RandomForestClassifier(n_estimators=100, random_state=int(random_state), n_jobs=-1)
                    base_model.fit(X, y)
                    sfm = SelectFromModel(base_model, prefit=True, threshold="mean")
                    X_selected = sfm.transform(X)
                    selected_features = X.columns[sfm.get_support()].tolist()
                    st.write(f"**Selected Features ({len(selected_features)}):**", selected_features)

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y, test_size=float(test_size), random_state=int(random_state), stratify=y
                    )
                    model = RandomForestClassifier(
                        n_estimators=int(rf_estimators),
                        max_depth=int(max_depth),
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=int(random_state),
                        n_jobs=-1
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    # Cross-validation
                    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')
                    st.write("**Cross-Validation Accuracy Scores:**", [f"{s:.4f}" for s in cv_scores])
                    st.write("**Average CV Accuracy:**", f"{cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

                    # Metrics
                    acc = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_proba)
                    st.write("**Accuracy:**", f"{acc:.4f}")
                    st.write("**ROC AUC:**", f"{auc:.4f}")
                    st.text("Classification Report:")
                    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', labels=[0, 1], zero_division=0)
                    rep_df = pd.DataFrame({"precision": [pr], "recall": [rc], "f1": [f1]}, index=["weighted_avg"])
                    st.dataframe(rep_df)

                    # Display visualizations in container
                    with viz_container:
                        # Confusion matrix plot
                        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                        fig_cm = plt.figure()
                        plt.imshow(cm, interpolation='nearest')
                        plt.title('Confusion Matrix')
                        plt.colorbar()
                        tick_marks = np.arange(2)
                        plt.xticks(tick_marks, ['0', '1'])
                        plt.yticks(tick_marks, ['0', '1'])
                        thresh = cm.max() / 2.0
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                                         color="white" if cm[i, j] > thresh else "black")
                        plt.ylabel('Actual')
                        plt.xlabel('Predicted')
                        st.pyplot(fig_cm, use_container_width=True)

                        # ROC curve
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        fig_roc = plt.figure()
                        plt.plot(fpr, tpr, label="ROC")
                        plt.plot([0, 1], [0, 1], linestyle='--')
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.title("ROC Curve")
                        plt.legend()
                        st.pyplot(fig_roc, use_container_width=True)

                        # Feature importances
                        importances = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)[:20]
                        st.write("**Top 20 Feature Importances**")
                        fig_imp = plt.figure()
                        importances[::-1].plot(kind='barh')
                        plt.xlabel('Importance')
                        plt.tight_layout()
                        st.pyplot(fig_imp, use_container_width=True)

                    # Allow download of predictions
                    out = pd.DataFrame(X_test, columns=selected_features)
                    out[target_col + "_true"] = y_test.values
                    out["predicted"] = y_pred
                    out["prob_fraud"] = y_proba
                    st.download_button(
                        "Download test predictions",
                        out.to_csv(index=False).encode('utf-8'),
                        file_name="rf_test_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"RandomForest training failed: {str(e)} - Check data or parameters.")

    st.divider()

    # ---------------- Unsupervised Models: IsolationForest & One-Class SVM ----------------
    st.subheader("Unsupervised Anomaly Detection")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for unsupervised detection. Consider encoding/cleaning your data.")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            unsup_model_choice = st.selectbox("Model", ["Isolation Forest", "One-Class SVM"])
        with col2:
            contamination = st.slider("Contamination (expected fraud rate)", 0.001, 0.2, 0.02, 0.001)
        with col3:
            feature_choices = st.multiselect("Features", numeric_cols, default=numeric_cols[: min(5, len(numeric_cols))])

        if st.button("Run Unsupervised Detection", use_container_width=True):
            try:
                data = df[feature_choices].dropna()
                if data.shape[0] < 10 or data.shape[1] == 0:
                    st.error("Not enough data after dropping NaNs. Please select valid features or clean data.")
                else:
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data)

                    if unsup_model_choice == "Isolation Forest":
                        model = IsolationForest(contamination=float(contamination), random_state=42)
                    else:
                        model = OneClassSVM(kernel="rbf", gamma="scale", nu=float(contamination))

                    preds = model.fit_predict(data_scaled)
                    result = df.copy()
                    result.loc[data.index, "Anomaly"] = np.where(preds == -1, "Fraud", "Normal")

                    st.dataframe(result.head(50), use_container_width=True)

                    with viz_container:
                        if len(feature_choices) >= 2:
                            fig_sc = plt.figure()
                            normal = data[preds == 1]
                            fraud = data[preds == -1]
                            plt.scatter(normal.iloc[:, 0], normal.iloc[:, 1], alpha=0.5, label="Normal")
                            plt.scatter(fraud.iloc[:, 0], fraud.iloc[:, 1], alpha=0.7, label="Fraud")
                            plt.xlabel(feature_choices[0])
                            plt.ylabel(feature_choices[1])
                            plt.legend()
                            plt.title("Anomaly Scatter (first 2 features)")
                            st.pyplot(fig_sc, use_container_width=True)
                        else:
                            fig_h = plt.figure()
                            plt.hist(data.values.ravel(), bins=30)
                            plt.title(f"Distribution: {feature_choices[0] if feature_choices else ''}")
                            st.pyplot(fig_h, use_container_width=True)

                    st.download_button(
                        "Download Unsupervised Results",
                        result.to_csv(index=False).encode("utf-8"),
                        file_name="unsupervised_fraud_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Unsupervised detection failed: {e}")

    st.divider()

    # ---------------- Network Graph Visualization ----------------
    st.subheader("Transaction Network Graph")
    st.caption("Builds a simple bipartite network if applicable (e.g., user ⇄ device/ip).")

    node_left = st.selectbox("Left node (e.g., customer/account/id)", ["<Auto>"] + list(df.columns))
    node_right = st.selectbox("Right node (e.g., device/ip/merchant)", ["<Auto>"] + list(df.columns))

    if st.button("Generate Network Graph", use_container_width=True):
        try:
            left_candidates = ["account_id", "customer_id", "user_id", "id"]
            right_candidates = ["device_hash", "device_id", "ip_address", "merchant_id", "merchant"]

            left = node_left if node_left != "<Auto>" else next((c for c in left_candidates if c in df.columns), None)
            right = node_right if node_right != "<Auto>" else next((c for c in right_candidates if c in df.columns), None)

            if not left or not right:
                st.error("Please select valid columns for left and right nodes.")
            else:
                pairs = df[[left, right]].dropna().astype(str).head(5000)
                G = nx.Graph()
                edges = list(zip(pairs[left], pairs[right]))
                G.add_edges_from(edges)

                with viz_container:
                    fig_g = plt.figure(figsize=(8, 6))
                    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
                    nx.draw_networkx_nodes(G, pos, node_size=30)
                    nx.draw_networkx_edges(G, pos, alpha=0.3)
                    plt.axis('off')
                    plt.title(f"Network: {left} ↔ {right} (n={G.number_of_nodes()} nodes, e={G.number_of_edges()} edges)")
                    st.pyplot(fig_g, use_container_width=True)
        except Exception as e:
            st.error(f"Network graph generation failed: {e}")

else:
    st.info("Load data from the sidebar to begin.")