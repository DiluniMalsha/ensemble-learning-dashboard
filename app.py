"""
Ensemble Learning Dashboard
===========================
Upload a CSV -> train individual models and ensembles (Bagging, Random Forest, AdaBoost,
Gradient Boosting, XGBoost, CatBoost, Voting, Stacking) -> compare them -> make predictions.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ml_core import (  # noqa: E402
    ENSEMBLE_MODELS, HAS_CATBOOST, INDIVIDUAL_MODELS, MODEL_INFO, detect_columns,
    feature_importance, prepare_xy, stacking_meta_weights, train_and_evaluate,
)

DATA_DIR = ROOT / "data"
SAMPLES = {
    "📧 SMS Spam (real, UCI SMS Spam Collection)": ("sms_spam.csv", "label"),
    "🌧️ Sri Lanka Rain Tomorrow (simulated local weather)": ("sri_lanka_rain.csv", "rain_tomorrow"),
}
METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
PALETTE = {"Individual": "#8C9BAB", "Ensemble": "#2A7FFF"}

st.set_page_config(page_title="Ensemble Learning Dashboard", page_icon="🌲", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
.small-note {color: #6b7280; font-size: 0.86rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------------------
def reset_results():
    st.session_state["results"] = {}


if "results" not in st.session_state:
    reset_results()


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes | None, sample_name: str | None) -> pd.DataFrame:
    if file_bytes is not None:
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_csv(DATA_DIR / sample_name)


# --------------------------------------------------------------------------------------
# Sidebar - data & split settings
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("1️⃣ Data")
    source = st.radio("Data source", list(SAMPLES) + ["⬆️ Upload my own CSV"], index=0)
    default_target = None
    if source.startswith("⬆️"):
        up = st.file_uploader("CSV file (one row = one sample, one column = target label)", type=["csv"])
        if up is None:
            st.info("Upload a CSV to continue, or choose a sample dataset.")
            st.stop()
        df = load_csv(up.getvalue(), None)
        data_key = f"upload:{up.name}:{len(up.getvalue())}"
    else:
        fname, default_target = SAMPLES[source]
        df = load_csv(None, fname)
        data_key = f"sample:{fname}"

    cols = list(df.columns)
    target = st.selectbox("🎯 Target column (what to predict)", cols,
                          index=cols.index(default_target) if default_target in cols else len(cols) - 1)
    features = st.multiselect("Feature columns", [c for c in cols if c != target],
                              default=[c for c in cols if c != target])
    st.header("2️⃣ Evaluation")
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.select_slider("Cross-validation folds (0 = off, slower if on)", [0, 3, 5], value=0)
    seed = st.number_input("Random seed", value=42, step=1)

    setup_key = (data_key, target, tuple(features), test_size, seed)
    if st.session_state.get("setup_key") != setup_key:
        st.session_state["setup_key"] = setup_key
        reset_results()

    st.divider()
    if st.button("🗑️ Clear trained models", width="stretch"):
        reset_results()
    trained_count = st.empty()

if not features:
    st.warning("Select at least one feature column.")
    st.stop()

# --------------------------------------------------------------------------------------
# Prepare data (shared by all tabs)
# --------------------------------------------------------------------------------------
X, y, label_enc = prepare_xy(df, target, features)
classes = list(label_enc.classes_)
n_classes = len(classes)
if n_classes < 2:
    st.error("The target column needs at least 2 classes.")
    st.stop()
if n_classes > 20:
    st.error(f"Target has {n_classes} unique values - this looks like a regression target. "
             "This dashboard is for classification; pick a categorical target.")
    st.stop()
plan = detect_columns(X)
strat = y if np.bincount(y).min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(seed),
                                                    stratify=strat)

st.title("🌲 Ensemble Learning Dashboard")
st.markdown("<span class='small-note'>Individual models vs. Bagging, Boosting (RF, AdaBoost, XGBoost, CatBoost) "
            "and Stacking — train, compare and predict on your own data.</span>", unsafe_allow_html=True)

tab_data, tab_ind, tab_ens, tab_cmp, tab_pred, tab_ts = st.tabs([
    "📂 Data", "🧩 Individual Models", "🤝 Ensembles", "📊 Comparison", "🔮 Predict", "📈 Hybrid ARIMA+RF (bonus)"])


# --------------------------------------------------------------------------------------
# Reusable: hyper-parameter widgets + training
# --------------------------------------------------------------------------------------
def param_widgets(name: str, key: str) -> dict:
    p = {}
    tree_like = name in ("Decision Tree", "Bagging (Decision Trees)", "Random Forest", "Gradient Boosting",
                         "XGBoost", "CatBoost")
    if name in ("Bagging (Decision Trees)", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost",
                "CatBoost", "Voting (soft)", "Stacking"):
        p["n_estimators"] = st.slider("n_estimators (number of trees / rounds)", 20, 500, 200, 10, key=f"{key}n")
    if tree_like:
        d = st.slider("max_depth (0 = default/unlimited)", 0, 20, 0, 1, key=f"{key}d")
        p["max_depth"] = d or None
    if name in ("AdaBoost", "Gradient Boosting", "XGBoost", "CatBoost"):
        default_lr = 1.0 if name == "AdaBoost" else 0.1
        p["learning_rate"] = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                                              value=default_lr, key=f"{key}lr")
    if name in ("Logistic Regression", "SVM"):
        p["C"] = st.select_slider("C (inverse regularisation)", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0,
                                  key=f"{key}c")
    if name == "KNN":
        p["n_neighbors"] = st.slider("k (neighbours)", 1, 31, 7, 2, key=f"{key}k")
    return p


def train_models(names: list[str], params: dict[str, dict]):
    prog = st.progress(0.0, text="Training…")
    for i, n in enumerate(names):
        prog.progress(i / len(names), text=f"Training **{n}** …")
        try:
            res = train_and_evaluate(n, plan, X_train, y_train, X_test, y_test, n_classes,
                                     params.get(n), cv_folds=cv_folds)
            st.session_state["results"][n] = res
        except Exception as e:  # keep the app alive, show the reason
            st.error(f"{n} failed: {e}")
    prog.progress(1.0, text="Done ✅")


def results_table(names: list[str] | None = None) -> pd.DataFrame:
    res = st.session_state["results"]
    rows = []
    for n, r in res.items():
        if names is None or n in names:
            rows.append({"Model": n, "Type": r.family, **r.metrics})
    return pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()


def show_table(tbl: pd.DataFrame):
    fmt = {c: "{:.4f}" for c in tbl.columns if c not in ("Type", "Train time (s)")}
    fmt["Train time (s)"] = "{:.2f}"
    hi = [c for c in METRIC_COLS + ["CV F1 mean"] if c in tbl.columns]
    st.dataframe(tbl.style.format(fmt, na_rep="–").highlight_max(subset=hi, color="#c8f7c5"),
                 width="stretch")


def confusion_fig(r, title=None):
    fig = px.imshow(r.cm, x=[f"pred {c}" for c in classes], y=[f"true {c}" for c in classes],
                    text_auto=True, color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(title=title or r.name, height=300, margin=dict(l=10, r=10, t=40, b=10),
                      coloraxis_showscale=False)
    return fig


def training_tab(model_names: list[str], key: str, groups: dict[str, list[str]] | None = None):
    st.subheader("Choose models & hyper-parameters")
    available = [m for m in model_names if not (m == "CatBoost" and (not HAS_CATBOOST or plan.is_text))]
    if "CatBoost" in model_names and "CatBoost" not in available:
        st.caption("ℹ️ CatBoost is hidden: it needs categorical/numeric data (not free text) and the `catboost` package.")
    if groups:
        picked = []
        cols = st.columns(len(groups))
        for col, (g, ms) in zip(cols, groups.items()):
            with col:
                st.markdown(f"**{g}**")
                for m in ms:
                    if m in available and st.checkbox(m, value=m in ("Random Forest", "XGBoost", "Stacking"),
                                                      key=f"{key}chk{m}", help=MODEL_INFO[m]):
                        picked.append(m)
    else:
        picked = st.multiselect("Models", available, default=available, key=f"{key}ms")
    params = {}
    if picked:
        with st.expander("⚙️ Hyper-parameters (optional)"):
            pcols = st.columns(min(3, len(picked)))
            for i, m in enumerate(picked):
                with pcols[i % len(pcols)]:
                    st.markdown(f"**{m}**")
                    params[m] = param_widgets(m, f"{key}{m}")
    if st.button(f"🚀 Train {len(picked)} model(s)", type="primary", disabled=not picked, key=f"{key}btn"):
        train_models(picked, params)

    tbl = results_table(model_names)
    if not tbl.empty:
        st.subheader("Results on the test set")
        show_table(tbl)
        trained = [n for n in model_names if n in st.session_state["results"]]
        cols = st.columns(min(3, len(trained)))
        for i, n in enumerate(trained):
            with cols[i % len(cols)]:
                st.plotly_chart(confusion_fig(st.session_state["results"][n]), width="stretch",
                                key=f"{key}cm{n}")


# --------------------------------------------------------------------------------------
# TAB 1 - Data
# --------------------------------------------------------------------------------------
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features used", len(features))
    c3.metric("Classes", n_classes)
    c4.metric("Train / Test", f"{len(X_train):,} / {len(X_test):,}")

    st.markdown(f"**Detected column types** → numeric: `{plan.numeric or '–'}` · categorical: "
                f"`{plan.categorical or '–'}` · free text (TF-IDF): `{plan.text or '–'}`")
    if plan.is_text:
        st.info("Free-text column detected → the text is converted to TF-IDF word/bigram features "
                "before training (like extracting keywords from emails).")
    left, right = st.columns([2, 1])
    with left:
        st.dataframe(df.head(200), width="stretch", height=330)
    with right:
        counts = df[target].astype(str).value_counts().reset_index()
        counts.columns = [target, "count"]
        fig = px.bar(counts, x=target, y="count", text="count", title="Class balance",
                     color_discrete_sequence=["#2A7FFF"])
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        imb = counts["count"].max() / counts["count"].min()
        if imb > 3:
            st.warning(f"Imbalanced classes (ratio {imb:.1f}:1) – look at **F1 / Recall**, not only Accuracy.")
    miss = df[features].isna().sum()
    if miss.sum():
        st.caption("Missing values (filled automatically: median for numeric, most-frequent for categorical): "
                   + ", ".join(f"{k}: {v}" for k, v in miss[miss > 0].items()))
    if plan.numeric:
        with st.expander("📈 Numeric feature distributions"):
            col = st.selectbox("Feature", plan.numeric)
            st.plotly_chart(px.histogram(df, x=col, color=df[target].astype(str), barmode="overlay",
                                         nbins=40, opacity=0.65), width="stretch")

# --------------------------------------------------------------------------------------
# TAB 2 - Individual models
# --------------------------------------------------------------------------------------
with tab_ind:
    st.markdown("Single (\"weak\" or classical) learners – the **baseline** the ensembles must beat.")
    training_tab(INDIVIDUAL_MODELS, "ind")

# --------------------------------------------------------------------------------------
# TAB 3 - Ensembles
# --------------------------------------------------------------------------------------
with tab_ens:
    st.markdown("**Bagging** trains models in *parallel* on bootstrap samples (reduces variance). "
                "**Boosting** trains models *sequentially*, each fixing the previous one's errors (reduces bias). "
                "**Stacking** trains a meta-model to combine different base models.")
    training_tab(ENSEMBLE_MODELS, "ens", groups={
        "Bagging": ["Bagging (Decision Trees)", "Random Forest"],
        "Boosting": ["AdaBoost", "Gradient Boosting", "XGBoost", "CatBoost"],
        "Combining": ["Voting (soft)", "Stacking"],
    })
    if "Stacking" in st.session_state["results"]:
        w = stacking_meta_weights(st.session_state["results"]["Stacking"].pipeline)
        if w is not None:
            st.markdown("**Inside the Stacking model** – how much the meta-learner (Logistic Regression) "
                        "trusts each base model's probability:")
            st.plotly_chart(px.bar(w, x="base model", y="meta weight", text_auto=".2f", height=280,
                                   color_discrete_sequence=["#2A7FFF"]), width="stretch")

# --------------------------------------------------------------------------------------
# TAB 4 - Comparison dashboard
# --------------------------------------------------------------------------------------
with tab_cmp:
    tbl = results_table()
    if tbl.empty:
        st.info("Train some models in the **Individual Models** and **Ensembles** tabs first. "
                "Tip: train *Logistic Regression, Decision Tree, Random Forest, XGBoost, Stacking* for a good story.")
    else:
        main_metric = st.radio("Rank by", [m for m in METRIC_COLS if m in tbl.columns], index=3, horizontal=True)
        tbl = tbl.sort_values(main_metric, ascending=False)
        best = tbl.index[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏆 Best model", best)
        k2.metric(f"Best {main_metric}", f"{tbl.iloc[0][main_metric]:.4f}")
        ind, ens = tbl[tbl.Type == "Individual"], tbl[tbl.Type == "Ensemble"]
        if len(ind) and len(ens):
            gain = ens[main_metric].max() - ind[main_metric].max()
            k3.metric("Best ensemble vs best single", f"{gain:+.4f}",
                      help="Positive = ensembles beat the best individual model")
        fastest = tbl["Train time (s)"].idxmin()
        k4.metric("⚡ Fastest to train", fastest, f"{tbl.loc[fastest, 'Train time (s)']:.2f}s", delta_color="off")

        show_table(tbl)

        long = tbl.reset_index().melt(id_vars=["Model", "Type"],
                                      value_vars=[m for m in METRIC_COLS if m in tbl.columns],
                                      var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", height=420,
                     title="All metrics by model")
        lo = max(0.0, long["Score"].min() - 0.05)
        fig.update_yaxes(range=[lo, 1.0])
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            sdf = tbl.reset_index()
            sdf["label"] = [m if i < 3 else "" for i, m in enumerate(sdf["Model"])]  # label top-3 only
            sc = px.scatter(sdf, x="Train time (s)", y=main_metric, color="Type", text="label", hover_name="Model",
                            color_discrete_map=PALETTE, title=f"{main_metric} vs training cost (hover for names)",
                            log_x=True, height=420)
            sc.update_traces(textposition="top center", marker_size=12)
            st.plotly_chart(sc, width="stretch")
        with c2:
            roc_models = [n for n in tbl.index if st.session_state["results"][n].roc is not None]
            if roc_models:
                rf = go.Figure()
                for n in roc_models:
                    fpr, tpr = st.session_state["results"][n].roc
                    rf.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                            name=f"{n} (AUC {tbl.loc[n, 'ROC-AUC']:.3f})"))
                rf.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"),
                                        showlegend=False))
                rf.update_layout(title="ROC curves", xaxis_title="False positive rate",
                                 yaxis_title="True positive rate", height=420, legend=dict(font_size=10))
                st.plotly_chart(rf, width="stretch")
            else:
                st.caption("ROC curves are shown for binary targets.")

        if "CV F1 mean" in tbl.columns and tbl["CV F1 mean"].notna().any():
            cvt = tbl.dropna(subset=["CV F1 mean"]).reset_index()
            st.plotly_chart(px.bar(cvt, x="Model", y="CV F1 mean", error_y="CV F1 std", color="Type",
                                   color_discrete_map=PALETTE, title=f"{cv_folds}-fold cross-validation F1 "
                                   "(error bar = std → stability)", height=360), width="stretch")

        st.subheader("🔍 What drives the predictions?")
        fi_model = st.selectbox("Feature importance for", list(tbl.index))
        fi = feature_importance(st.session_state["results"][fi_model].pipeline)
        if fi is not None:
            st.plotly_chart(px.bar(fi.iloc[::-1], x="importance", y="feature", orientation="h", height=450,
                                   color_discrete_sequence=["#2A7FFF"]), width="stretch")
        else:
            st.caption("This model type has no built-in feature importance (e.g. KNN, Naive Bayes, Voting, Stacking).")

        st.subheader("🧠 Automatic interpretation")
        notes = [f"**{best}** gives the best {main_metric} ({tbl.iloc[0][main_metric]:.4f})."]
        if len(ind) and len(ens):
            bi, be = ind[main_metric].idxmax(), ens[main_metric].idxmax()
            g = ens[main_metric].max() - ind[main_metric].max()
            notes.append(f"Best ensemble **{be}** vs best single model **{bi}**: {g:+.4f} {main_metric}. "
                         + ("Ensembling clearly helps here." if g > 0.01 else
                            "The gain is small – the data may be (almost) linearly separable or noisy, "
                            "so a simple model is already close to the ceiling."))
        if "Decision Tree" in tbl.index and "Random Forest" in tbl.index:
            g = tbl.loc["Random Forest", main_metric] - tbl.loc["Decision Tree", main_metric]
            notes.append(f"Random Forest vs a single Decision Tree: {g:+.4f} → bagging reduces the variance of trees.")
        if {"Random Forest", "XGBoost"} <= set(tbl.index):
            d = tbl.loc["XGBoost", main_metric] - tbl.loc["Random Forest", main_metric]
            notes.append(f"XGBoost vs Random Forest: {d:+.4f}. "
                         + ("Boosting wins (lower bias)." if d > 0.005 else
                            "Random Forest wins or ties – boosting needs tuning (learning_rate, depth) to shine."
                            if d < -0.005 else "Practically a tie."))
        if "Stacking" in tbl.index:
            others = tbl.drop(index="Stacking")
            if len(others):
                d = tbl.loc["Stacking", main_metric] - others[main_metric].max()
                notes.append(f"Stacking vs best non-stacking model: {d:+.4f}. Stacking costs "
                             f"{tbl.loc['Stacking', 'Train time (s)']:.1f}s to train (5-fold internal CV).")
        if n_classes == 2 and np.bincount(y).max() / np.bincount(y).min() > 3:
            notes.append("Classes are imbalanced → Accuracy is inflated; F1 and Recall are the fairer metrics.")
        for n_ in notes:
            st.markdown(f"- {n_}")

        st.download_button("⬇️ Download results (CSV)", tbl.to_csv().encode(), "model_comparison.csv", "text/csv")

# --------------------------------------------------------------------------------------
# TAB 5 - Predict / inference
# --------------------------------------------------------------------------------------
with tab_pred:
    res = st.session_state["results"]
    if not res:
        st.info("Train at least one model first.")
    else:
        tbl = results_table().sort_values("F1", ascending=False)
        model_name = st.selectbox("Model to use", list(tbl.index),
                                  help="Sorted by F1 – the first one is your best model.")
        pipe = res[model_name].pipeline

        def show_prediction(row: pd.DataFrame):
            proba = pipe.predict_proba(row)[0]
            pred = classes[int(np.argmax(proba))]
            c1, c2 = st.columns([1, 2])
            c1.metric("Prediction", pred)
            c1.caption(f"Confidence: **{proba.max() * 100:.1f}%** · model: {model_name}")
            pf = pd.DataFrame({"class": classes, "probability": proba})
            pfig = px.bar(pf, x="probability", y="class", orientation="h", range_x=[0, 1], text_auto=".1%",
                          height=140 + 40 * len(classes), color_discrete_sequence=["#2A7FFF"],
                          title="Class probabilities")
            pfig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            pfig.update_yaxes(type="category")
            c2.plotly_chart(pfig, width="stretch")

        st.subheader("Single prediction")
        if plan.is_text:
            examples = {
                "— type your own —": "",
                "Prize scam": "Congratulations! You have WON a £1000 cash prize. Call 09061701461 now to claim, T&C apply",
                "Normal message": "Hey, are we still meeting for lunch at 1? I'll bring the notes from class",
                "Promo": "FREE entry to our weekly draw! Text WIN to 80086 to receive your ringtone. Stop txt STOP",
            }
            ex = st.selectbox("Example", list(examples))
            txt = st.text_area(f"Enter `{plan.text[0]}` text (e.g. paste an email/SMS)", examples[ex], height=110)
            row = X.iloc[[0]].copy()
            for c in row.columns:
                row[c] = txt if c in plan.text else X[c].mode().iloc[0]
            if st.button("🔮 Classify", type="primary") and txt.strip():
                show_prediction(row)
        else:
            with st.form("predict_form"):
                vals = {}
                cols = st.columns(3)
                for i, c in enumerate(X.columns):
                    with cols[i % 3]:
                        if c in plan.numeric:
                            s = X[c].dropna()
                            is_int = pd.api.types.is_integer_dtype(X[c])
                            vals[c] = st.number_input(c, value=int(s.median()) if is_int else float(s.median()),
                                                      step=1 if is_int else None,
                                                      help=f"range in data: {s.min():g} – {s.max():g}")
                        elif c in plan.categorical:
                            opts = sorted(X[c].dropna().astype(str).unique())
                            vals[c] = st.selectbox(c, opts)
                        else:
                            vals[c] = st.text_input(c, str(X[c].iloc[0]))
                go_btn = st.form_submit_button("🔮 Predict", type="primary")
            if go_btn:
                row = pd.DataFrame([vals])
                for c in plan.numeric:
                    row[c] = row[c].astype(X[c].dtype)
                show_prediction(row)

        st.subheader("Batch prediction (upload a CSV with the same feature columns)")
        bf = st.file_uploader("CSV to score", type=["csv"], key="batch")
        if bf is not None:
            new = pd.read_csv(bf)
            missing = [c for c in features if c not in new.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                pr = pipe.predict_proba(new[features])
                out = new.copy()
                out["prediction"] = [classes[i] for i in pr.argmax(1)]
                out["confidence"] = pr.max(1).round(4)
                st.dataframe(out.head(100), width="stretch")
                st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")

        st.download_button("💾 Download trained model (.pkl)",
                           pickle.dumps({"pipeline": pipe, "classes": classes, "features": features}),
                           f"{model_name.replace(' ', '_').lower()}.pkl",
                           help="Load later with pickle (keep the src/ folder on the Python path).")

# --------------------------------------------------------------------------------------
# TAB 6 - Hybrid ARIMA + RF (time series stacking)
# --------------------------------------------------------------------------------------
with tab_ts:
    st.markdown("""
**Hybrid SARIMA + Random Forest** – a simple *stacking* idea for time series (e.g. rainfall):
1. **SARIMA** captures the linear + seasonal pattern (monsoons).
2. Its **residuals** (errors) are given to a **Random Forest**, which tries to learn the non-linear part.
3. **Final forecast = SARIMA forecast + RF residual correction.**  All forecasts are one-step-ahead (next month).
""")
    ts_src = st.radio("Series", ["Colombo monthly rainfall (simulated sample)", "Upload CSV (date, value)"],
                      horizontal=True)
    series = None
    if ts_src.startswith("Upload"):
        tf = st.file_uploader("CSV with a date column and a numeric value column (monthly)", type=["csv"], key="ts")
        if tf is not None:
            tdf = pd.read_csv(tf)
            dcol = st.selectbox("Date column", tdf.columns)
            vcol = st.selectbox("Value column", [c for c in tdf.columns if c != dcol])
            series = tdf.assign(**{dcol: pd.to_datetime(tdf[dcol])}).set_index(dcol)[vcol].sort_index()
    else:
        series = pd.read_csv(DATA_DIR / "colombo_monthly_rainfall.csv", parse_dates=["month"]) \
            .set_index("month")["rainfall_mm"]
    if series is not None:
        c1, c2 = st.columns(2)
        horizon = c1.slider("Test period (last N months)", 12, 72, 36, 6)
        n_est = c2.slider("RF trees", 100, 600, 300, 50)
        if st.button("📈 Run SARIMA vs RF vs Hybrid", type="primary"):
            from hybrid_ts import run_hybrid
            with st.spinner("Fitting SARIMA(1,0,1)(1,1,1,12) and Random Forests…"):
                try:
                    preds, scores, _ = run_hybrid(series, test_size=horizon, n_estimators=n_est)
                    st.session_state["ts_out"] = (preds, scores)
                except Exception as e:
                    st.error(f"Failed: {e}. The series needs to be monthly with at least ~5 years of data.")
        if "ts_out" in st.session_state:
            preds, scores = st.session_state["ts_out"]
            best_ts = scores["RMSE"].idxmin()
            st.dataframe(scores.style.format("{:.2f}").highlight_min(color="#c8f7c5"), width="stretch")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index[-(len(preds) + 48):], y=series.iloc[-(len(preds) + 48):],
                                     name="Actual", line=dict(color="black", width=2)))
            for col, color in zip(["SARIMA", "Random Forest", "Hybrid SARIMA+RF"], ["#F59E0B", "#10B981", "#2A7FFF"]):
                fig.add_trace(go.Scatter(x=preds.index, y=preds[col], name=col, line=dict(color=color, dash="dot")))
            fig.update_layout(height=430, title="Test-period one-step-ahead forecasts", yaxis_title="value")
            st.plotly_chart(fig, width="stretch")
            st.markdown(f"- Lowest RMSE: **{best_ts}**.\n"
                        "- The hybrid only helps when SARIMA's residuals still contain a *learnable, non-linear* "
                        "pattern; if they are mostly random noise, adding the RF gives little or no gain "
                        "(and can slightly overfit).\n"
                        "- RF alone cannot extrapolate trends and only sees lags, so it usually trails SARIMA on "
                        "strongly seasonal series.")

trained_count.caption(f"Trained so far: **{len(st.session_state['results'])}** model(s)")
