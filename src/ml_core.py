"""
Core ML logic shared by the Streamlit dashboard and the notebook.

- builds a preprocessing pipeline automatically (numeric / categorical / free-text columns)
- defines individual models and ensemble models (bagging, boosting, stacking, voting)
- trains + evaluates any model and returns comparable metrics
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
    RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:  # CatBoost is optional - the app still works without it
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:  # pragma: no cover
    HAS_CATBOOST = False

RANDOM_STATE = 42


# --------------------------------------------------------------------------------------
# Column detection & preprocessing
# --------------------------------------------------------------------------------------
@dataclass
class ColumnPlan:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    @property
    def is_text(self) -> bool:
        return len(self.text) > 0


def detect_columns(X: pd.DataFrame, max_cat_unique: int = 30) -> ColumnPlan:
    """Numeric -> numeric. Object with few unique values -> categorical.
    Object with long, mostly-unique strings -> free text (TF-IDF)."""
    plan = ColumnPlan()
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            plan.numeric.append(c)
        else:
            s = s.astype(str)
            nunique = s.nunique()
            avg_words = s.str.split().str.len().mean()
            if nunique > max_cat_unique and avg_words >= 3:
                plan.text.append(c)
            elif nunique <= max(max_cat_unique, 50):
                plan.categorical.append(c)
            # else: high-cardinality IDs etc. are dropped
    return plan


def _to_1d(x):
    """TfidfVectorizer needs a 1-D array of strings."""
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return np.asarray(x).astype(str).ravel()


def build_preprocessor(plan: ColumnPlan, max_text_features: int = 3000) -> ColumnTransformer:
    transformers = []
    if plan.numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), plan.numeric))
    if plan.categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), plan.categorical))
    for i, col in enumerate(plan.text):
        transformers.append((f"txt{i}", Pipeline([
            ("to1d", FunctionTransformer(_to_1d, feature_names_out="one-to-one")),
            ("tfidf", TfidfVectorizer(max_features=max_text_features, ngram_range=(1, 2),
                                      sublinear_tf=True, stop_words="english")),
        ]), [col]))
    # text -> sparse output; tabular -> dense (keeps SVM/KNN/NB simple)
    return ColumnTransformer(transformers, sparse_threshold=0.3 if plan.is_text else 0.0)


def prepare_xy(df: pd.DataFrame, target: str, feature_cols: list[str] | None = None):
    df = df.dropna(subset=[target]).copy()
    X = df[feature_cols] if feature_cols else df.drop(columns=[target])
    le = LabelEncoder()
    y = le.fit_transform(df[target].astype(str))
    return X, y, le


# --------------------------------------------------------------------------------------
# Model catalogue
# --------------------------------------------------------------------------------------
INDIVIDUAL_MODELS = ["Logistic Regression", "Decision Tree", "SVM", "KNN", "Naive Bayes"]
ENSEMBLE_MODELS = ["Bagging (Decision Trees)", "Random Forest", "AdaBoost", "Gradient Boosting",
                   "XGBoost", "CatBoost", "Voting (soft)", "Stacking"]

MODEL_INFO = {
    "Logistic Regression": "Linear model; fast, interpretable baseline.",
    "Decision Tree": "Single tree - the 'weak learner' that bagging/boosting combine.",
    "SVM": "Max-margin classifier (RBF kernel on tabular, linear kernel on text).",
    "KNN": "Votes of the k nearest training points.",
    "Naive Bayes": "Probabilistic model assuming feature independence (Multinomial for text).",
    "Bagging (Decision Trees)": "Bootstrap aggregation: many trees on resampled data, majority vote. Reduces variance.",
    "Random Forest": "Bagging + random feature subsets at each split -> de-correlated trees.",
    "AdaBoost": "Sequential boosting; re-weights misclassified samples each round.",
    "Gradient Boosting": "Sequential trees fitting the gradient of the loss (sklearn implementation).",
    "XGBoost": "eXtreme Gradient Boosting: regularised, second-order, very fast boosting.",
    "CatBoost": "Gradient boosting with native categorical handling (ordered target statistics).",
    "Voting (soft)": "Averages predicted probabilities of LR, RF and XGBoost.",
    "Stacking": "Level-0: RF + XGBoost + SVM/NB; level-1 meta-learner (Logistic Regression) learns how to combine them.",
}


def _svm(text: bool, C: float):
    """SVM with calibrated probabilities (works on old and new scikit-learn versions)."""
    return CalibratedClassifierCV(SVC(kernel="linear" if text else "rbf", C=C, random_state=RANDOM_STATE),
                                  ensemble=False, cv=3)


def make_estimator(name: str, plan: ColumnPlan, n_classes: int, params: dict | None = None):
    """Return an (unfitted) classifier. `params` are the dashboard hyper-parameters."""
    p = params or {}
    text = plan.is_text
    n_est = int(p.get("n_estimators", 200))
    depth = p.get("max_depth")  # None = unlimited for trees
    lr = float(p.get("learning_rate", 0.1))

    xgb_kwargs = dict(n_estimators=n_est, max_depth=int(depth) if depth else 6, learning_rate=lr,
                      subsample=0.9, colsample_bytree=0.8, random_state=RANDOM_STATE,
                      n_jobs=-1, tree_method="hist", eval_metric="logloss" if n_classes == 2 else "mlogloss")

    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, C=float(p.get("C", 1.0)))
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=int(depth) if depth else None, random_state=RANDOM_STATE)
    if name == "SVM":
        return _svm(text, float(p.get("C", 1.0)))
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=int(p.get("n_neighbors", 7)))
    if name == "Naive Bayes":
        return MultinomialNB() if text else GaussianNB()
    if name == "Bagging (Decision Trees)":
        return BaggingClassifier(DecisionTreeClassifier(max_depth=int(depth) if depth else None),
                                 n_estimators=n_est, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=n_est, max_depth=int(depth) if depth else None,
                                      random_state=RANDOM_STATE, n_jobs=-1)
    if name == "AdaBoost":
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_est,
                                  learning_rate=float(p.get("learning_rate", 1.0)), random_state=RANDOM_STATE)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=n_est, max_depth=int(depth) if depth else 3,
                                          learning_rate=lr, random_state=RANDOM_STATE)
    if name == "XGBoost":
        return XGBClassifier(**xgb_kwargs)
    if name == "CatBoost":
        if not HAS_CATBOOST:
            raise ImportError("catboost is not installed (pip install catboost)")
        return CatBoostClassifier(iterations=n_est, depth=int(depth) if depth else 6, learning_rate=lr,
                                  random_seed=RANDOM_STATE, verbose=0, thread_count=-1,
                                  allow_writing_files=False)
    if name == "Voting (soft)":
        return VotingClassifier([
            ("lr", LogisticRegression(max_iter=2000)),
            ("rf", RandomForestClassifier(n_estimators=n_est, random_state=RANDOM_STATE, n_jobs=-1)),
            ("xgb", XGBClassifier(**xgb_kwargs)),
        ], voting="soft", n_jobs=1)
    if name == "Stacking":
        third = ("nb", MultinomialNB()) if text else ("svm", _svm(False, 1.0))
        return StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=n_est, random_state=RANDOM_STATE, n_jobs=-1)),
                ("xgb", XGBClassifier(**xgb_kwargs)),
                third,
            ],
            final_estimator=LogisticRegression(max_iter=2000),
            stack_method="predict_proba", cv=5, n_jobs=1, passthrough=False,
        )
    raise ValueError(f"Unknown model: {name}")


def build_catboost_preprocessor(plan: ColumnPlan) -> ColumnTransformer:
    """CatBoost handles categorical columns natively, so we do NOT one-hot encode them:
    numeric -> median impute, categorical -> kept as strings (CatBoost encodes them itself)."""
    transformers = []
    if plan.numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), plan.numeric))
    if plan.categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("str", FunctionTransformer(_as_str, feature_names_out="one-to-one")),
        ]), plan.categorical))
    return ColumnTransformer(transformers, sparse_threshold=0.0)


def _as_str(x):
    return np.asarray(x).astype(str)


def make_pipeline(name: str, plan: ColumnPlan, n_classes: int, params: dict | None = None) -> Pipeline:
    if name == "CatBoost":
        if plan.is_text:
            raise ValueError("CatBoost is disabled for free-text data in this dashboard (TF-IDF is too wide). "
                             "Use XGBoost instead.")
        clf = make_estimator(name, plan, n_classes, params)
        n_num = len(plan.numeric)
        clf.set_params(cat_features=list(range(n_num, n_num + len(plan.categorical))))
        return Pipeline([("prep", build_catboost_preprocessor(plan)), ("clf", clf)])
    return Pipeline([("prep", build_preprocessor(plan)), ("clf", make_estimator(name, plan, n_classes, params))])


# --------------------------------------------------------------------------------------
# Training & evaluation
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    family: str
    pipeline: Pipeline
    metrics: dict
    cm: np.ndarray
    roc: tuple | None
    y_proba: np.ndarray | None
    train_time: float


def evaluate(pipe: Pipeline, X_test, y_test, n_classes: int):
    y_pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
    avg = "binary" if n_classes == 2 else "weighted"
    m = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=avg, zero_division=0),
    }
    roc = None
    if proba is not None:
        if n_classes == 2:
            m["ROC-AUC"] = roc_auc_score(y_test, proba[:, 1])
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            roc = (fpr, tpr)
        else:
            m["ROC-AUC"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
    return m, confusion_matrix(y_test, y_pred), roc, proba


def train_and_evaluate(name, plan, X_train, y_train, X_test, y_test, n_classes, params=None,
                       cv_folds: int = 0) -> Result:
    pipe = make_pipeline(name, plan, n_classes, params)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    metrics, cm, roc, proba = evaluate(pipe, X_test, y_test, n_classes)
    metrics["Train time (s)"] = train_time
    if cv_folds and cv_folds > 1:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(make_pipeline(name, plan, n_classes, params), X_train, y_train,
                                 cv=cv, scoring="f1" if n_classes == 2 else "f1_weighted", n_jobs=1)
        metrics["CV F1 mean"] = scores.mean()
        metrics["CV F1 std"] = scores.std()
    family = "Individual" if name in INDIVIDUAL_MODELS else "Ensemble"
    return Result(name, family, pipe, metrics, cm, roc, proba, train_time)


# --------------------------------------------------------------------------------------
# Feature importance (tree models)
# --------------------------------------------------------------------------------------
def feature_names(pipe: Pipeline) -> list[str]:
    try:
        return list(pipe.named_steps["prep"].get_feature_names_out())
    except Exception:
        return []


def feature_importance(pipe: Pipeline, top: int = 15) -> pd.DataFrame | None:
    clf = pipe.named_steps["clf"]
    names = feature_names(pipe)
    imp = None
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_)
    elif hasattr(clf, "coef_"):
        coef = clf.coef_.toarray() if hasattr(clf.coef_, "toarray") else np.asarray(clf.coef_)
        imp = np.abs(coef).mean(axis=0)
    elif isinstance(clf, BaggingClassifier):
        imp = np.mean([e.feature_importances_ for e in clf.estimators_], axis=0)
    if imp is None or len(names) != len(imp):
        return None
    clean = [n.split("__", 1)[-1] for n in names]
    return (pd.DataFrame({"feature": clean, "importance": imp})
            .sort_values("importance", ascending=False).head(top))


def stacking_meta_weights(pipe: Pipeline) -> pd.DataFrame | None:
    """How much the meta-learner trusts each base model (Logistic Regression coefficients)."""
    clf = pipe.named_steps["clf"]
    if not isinstance(clf, StackingClassifier):
        return None
    coef = clf.final_estimator_.coef_
    names = [n for n, _ in clf.estimators]
    if coef.shape[1] == len(names):  # binary: one prob column per base model
        return pd.DataFrame({"base model": names, "meta weight": coef[0]})
    return None
