import re
from pathlib import Path
from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
)

st.set_page_config(page_title="Depression Text Data Product", layout="wide")


def find_default_csv():
    search_paths = [Path("."), Path("Depression dataset")]

    for folder in search_paths:
        if folder.exists():
            csv_files = list(folder.glob("*.csv"))
            if csv_files:
                return csv_files[0]

    return None


def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file), "Uploaded file"

        default_csv = find_default_csv()
        if default_csv is not None:
            return pd.read_csv(default_csv), str(default_csv)

        return None, None

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def get_default_column(columns, preferred_name):
    if preferred_name in columns:
        return list(columns).index(preferred_name)
    return 0


def clean_text_series(series, lowercase=True, punctuation=True, numbers=True):
    """
    Original preprocessing workflow used for the main model:
    lowercase text, remove punctuation, and remove numbers.
    """
    cleaned = series.astype(str)

    if lowercase:
        cleaned = cleaned.str.lower()

    if punctuation:
        cleaned = cleaned.apply(lambda x: re.sub(r"[^\w\s]", "", x))

    if numbers:
        cleaned = cleaned.apply(lambda x: re.sub(r"\d+", "", x))

    return cleaned


def validate_model_inputs(df, text_column, label_column):
    if text_column == label_column:
        return False, "The text column and label column cannot be the same."

    if df[text_column].dropna().empty:
        return False, "The selected text column is empty."

    unique_labels = df[label_column].nunique()

    if unique_labels < 2:
        return False, "The label column must contain at least two classes."

    if unique_labels > 10:
        return False, (
            f"The selected label column has {unique_labels} unique values. "
            "This is probably not the correct label column. Please select `label`."
        )

    label_counts = df[label_column].value_counts()

    if label_counts.min() < 2:
        return False, (
            "At least one class has fewer than 2 records. "
            "Please check that the correct label column is selected."
        )

    return True, "Inputs are valid."


def fig_to_png_bytes(fig):
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300)
    img_buffer.seek(0)
    return img_buffer


def create_plots_zip():
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        plot_files = {
            "class_distribution_plot.png": st.session_state.get("class_distribution_fig"),
            "text_length_distribution_plot.png": st.session_state.get("text_length_fig"),
            "confusion_matrix_plot.png": st.session_state.get("confusion_matrix_fig"),
            "roc_curve_plot.png": st.session_state.get("roc_curve_fig"),
            "precision_recall_curve_plot.png": st.session_state.get("precision_recall_fig"),
            "threshold_analysis_plot.png": st.session_state.get("threshold_fig"),
            "calibration_curve_plot.png": st.session_state.get("calibration_fig"),
            "baseline_model_comparison_plot.png": st.session_state.get("baseline_fig"),
            "iterative_refinement_plot.png": st.session_state.get("refinement_fig"),
            "positive_feature_importance_plot.png": st.session_state.get("positive_feature_fig"),
            "negative_feature_importance_plot.png": st.session_state.get("negative_feature_fig"),
        }

        for filename, fig in plot_files.items():
            if fig is not None:
                img_buffer = fig_to_png_bytes(fig)
                zip_file.writestr(filename, img_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer


def get_positive_class(y):
    labels = sorted(pd.Series(y).dropna().unique())
    if 1 in labels:
        return 1
    return labels[-1]


def binary_y(y, positive_class):
    return np.array([1 if value == positive_class else 0 for value in y])


def get_positive_probability(model, X_test_tfidf, positive_class):
    class_list = list(model.classes_)
    positive_index = class_list.index(positive_class)
    return model.predict_proba(X_test_tfidf)[:, positive_index]


def calculate_advanced_metrics(y_test, y_pred, y_prob, positive_class):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    y_test_binary = binary_y(y_test, positive_class)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test_binary, y_prob),
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }


def evaluate_thresholds(y_test, y_prob, positive_class):
    y_test_binary = binary_y(y_test, positive_class)
    threshold_results = []

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        y_pred_threshold = np.where(y_prob >= threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_threshold).ravel()

        threshold_results.append({
            "Threshold": threshold,
            "Precision": precision_score(y_test_binary, y_pred_threshold, zero_division=0),
            "Recall": recall_score(y_test_binary, y_pred_threshold, zero_division=0),
            "F1 Score": f1_score(y_test_binary, y_pred_threshold, zero_division=0),
            "False Positives": fp,
            "False Negatives": fn
        })

    return pd.DataFrame(threshold_results)


def run_cross_validation(X, y, positive_class):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("model", LogisticRegression(max_iter=1000))
    ])

    scorer = make_scorer(f1_score, pos_label=positive_class, zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scorer
    )

    return scores


def compare_baseline_models(X_train, X_test, y_train, y_test, positive_class):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=15),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    results = []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    y_test_binary = binary_y(y_test, positive_class)

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        if hasattr(model, "predict_proba"):
            y_score = get_positive_probability(model, X_test_tfidf, positive_class)
        else:
            y_score = model.decision_function(X_test_tfidf)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test_binary, y_score)
        })

    return pd.DataFrame(results)


def compare_refinements(X_train, X_test, y_train, y_test, positive_class):
    refinements = {
        "Baseline: 5,000 unigrams": TfidfVectorizer(stop_words="english", max_features=5000),
        "Refinement 1: 10,000 unigrams": TfidfVectorizer(stop_words="english", max_features=10000),
        "Refinement 2: 10,000 unigrams + bigrams": TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )
    }

    results = []
    y_test_binary = binary_y(y_test, positive_class)

    for name, vectorizer in refinements.items():
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)
        y_prob = get_positive_probability(model, X_test_tfidf, positive_class)

        results.append({
            "Iteration": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, pos_label=positive_class, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test_binary, y_prob)
        })

    return pd.DataFrame(results)


def plot_feature_importance(model, vectorizer, top_n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    top_positive = importance_df.sort_values("Coefficient", ascending=False).head(top_n)
    top_negative = importance_df.sort_values("Coefficient", ascending=True).head(top_n)

    return top_positive, top_negative


def train_single_post_model():
    url = "https://raw.githubusercontent.com/usmaann/Depression_Severity_Dataset/main/Reddit_depression_dataset.csv"

    df_single = pd.read_csv(url)

    df_single["label_binary"] = df_single["label"].apply(
        lambda x: 0 if x == "minimum" else 1
    )

    df_single["clean_text"] = clean_text_series(df_single["text"])

    X = df_single["clean_text"]
    y = df_single["label_binary"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, y)

    return model, vectorizer


def generate_report(
    accuracy,
    report_text,
    cm,
    roc_auc_score=None,
    advanced_metrics=None,
    cv_scores=None,
    threshold_results=None,
    baseline_results=None,
    refinement_results=None
):
    advanced_text = ""
    if advanced_metrics is not None:
        advanced_text = f"""
Advanced Metrics:
{pd.DataFrame([advanced_metrics]).to_string(index=False)}
"""

    cv_text = ""
    if cv_scores is not None:
        cv_text = f"""
Cross-Validation Results:
Fold F1 Scores: {cv_scores}
Mean F1 Score: {np.mean(cv_scores)}
Standard Deviation: {np.std(cv_scores)}
"""

    threshold_text = ""
    if threshold_results is not None:
        threshold_text = f"""
Threshold Analysis:
{threshold_results.to_string(index=False)}
"""

    baseline_text = ""
    if baseline_results is not None:
        baseline_text = f"""
Baseline Model Comparison:
{baseline_results.to_string(index=False)}
"""

    refinement_text = ""
    if refinement_results is not None:
        refinement_text = f"""
Iterative Refinement Comparison:
{refinement_results.to_string(index=False)}
"""

    return f"""
Depression-Related Language Detection Data Product Report

Purpose:
This report summarizes the results of the NLP data product. The product uses text preprocessing,
TF-IDF vectorization, Logistic Regression, and single-post prediction to identify depression-related
language patterns in social media text.

Data Analysis Methods:
- Text preprocessing using lowercase conversion, punctuation removal, and number removal
- TF-IDF feature extraction
- Logistic Regression classification
- Single-post prediction using a separate post-level model
- Probability scoring for single-post predictions
- Accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC evaluation
- Precision-recall curve
- Sensitivity and specificity analysis
- Cross-validation
- Threshold analysis
- Calibration analysis
- Baseline model comparison
- Iterative refinement comparison
- Feature importance analysis
- Exportable plots in PNG and ZIP format

Accuracy:
{accuracy}

ROC-AUC:
{roc_auc_score}

Classification Report:
{report_text}

Confusion Matrix:
{cm}

{advanced_text}
{cv_text}
{threshold_text}
{baseline_text}
{refinement_text}

Security and Privacy:
- The app does not require names, usernames, passwords, or personal identifiers.
- The dataset is processed only during the active app session.
- User-entered text is analyzed only during the active app session.
- The app is intended for educational and research analysis only.
- The app is not a clinical diagnosis tool.
"""


st.title("Depression-Related Language Detection Data Product")

st.write(
    "This Streamlit app analyzes social media text using Natural Language Processing (NLP), "
    "TF-IDF feature extraction, and Logistic Regression to identify depression-related language patterns. "
    "This tool is for educational analysis only and is not a clinical diagnosis tool."
)

st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Choose a section:",
    [
        "Upload Data",
        "Explore Data",
        "Preprocess Text",
        "Run Model",
        "Predict New Post",
        "Generate Report",
        "Help"
    ]
)

st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Optional: Upload CSV file", type=["csv"])

df, data_source = load_data(uploaded_file)

if df is not None:
    st.sidebar.success(f"Dataset loaded: {data_source}")
else:
    st.sidebar.warning("No dataset found. Upload a CSV file or place one in the project folder.")


if section == "Upload Data":
    st.header("Upload Data")

    if df is not None:
        st.success("Dataset loaded successfully.")
        st.write("Data source:", data_source)
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.dataframe(df.head())
    else:
        st.info("Upload a CSV file or place your dataset CSV in the same folder as app.py.")


elif section == "Explore Data":
    st.header("Explore Data")

    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Missing Values")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        st.dataframe(missing)

        st.subheader("Class Distribution")

        if "label" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            df["label"].value_counts().sort_index().plot(kind="bar", ax=ax)
            ax.set_title("Class Distribution")
            ax.set_xlabel("Label")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            st.session_state["class_distribution_fig"] = fig

            st.download_button(
                label="Download Class Distribution Plot as PNG",
                data=fig_to_png_bytes(fig),
                file_name="class_distribution_plot.png",
                mime="image/png"
            )
        else:
            st.warning("No `label` column found.")

        st.subheader("Text Length Distribution")

        if "post_text" in df.columns:
            text_length = df["post_text"].astype(str).apply(len)
            fig, ax = plt.subplots(figsize=(8, 5))
            text_length.hist(bins=50, ax=ax)
            ax.set_title("Text Length Distribution")
            ax.set_xlabel("Number of Characters")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            st.session_state["text_length_fig"] = fig

            st.download_button(
                label="Download Text Length Distribution Plot as PNG",
                data=fig_to_png_bytes(fig),
                file_name="text_length_distribution_plot.png",
                mime="image/png"
            )

            if "label" in df.columns:
                st.subheader("Average Text Length by Class")
                length_df = pd.DataFrame({
                    "label": df["label"],
                    "text_length": text_length
                })
                avg_length = length_df.groupby("label")["text_length"].mean().reset_index()
                st.dataframe(avg_length)

                fig, ax = plt.subplots(figsize=(8, 5))
                avg_length.plot(kind="bar", x="label", y="text_length", legend=False, ax=ax)
                ax.set_title("Average Text Length by Class")
                ax.set_xlabel("Label")
                ax.set_ylabel("Average Number of Characters")
                st.pyplot(fig)

        else:
            st.warning("No `post_text` column found.")
    else:
        st.info("Please load a dataset first.")


elif section == "Preprocess Text":
    st.header("Preprocess Text")

    if df is not None:
        default_text_index = get_default_column(df.columns, "post_text")

        text_column = st.selectbox(
            "Select text column:",
            df.columns,
            index=default_text_index
        )

        lowercase = st.checkbox("Convert to lowercase", value=True)
        punctuation = st.checkbox("Remove punctuation", value=True)
        numbers = st.checkbox("Remove numbers", value=True)

        if st.button("Apply Cleaning"):
            df["clean_text"] = clean_text_series(
                df[text_column],
                lowercase=lowercase,
                punctuation=punctuation,
                numbers=numbers
            )

            st.success("Text preprocessing completed.")
            st.dataframe(df[[text_column, "clean_text"]].head())
    else:
        st.info("Please load a dataset first.")


elif section == "Run Model":
    st.header("Run Model")

    if df is not None:
        default_text_index = get_default_column(df.columns, "post_text")
        default_label_index = get_default_column(df.columns, "label")

        text_column = st.selectbox(
            "Select text column:",
            df.columns,
            index=default_text_index,
            key="model_text"
        )

        label_column = st.selectbox(
            "Select label column:",
            df.columns,
            index=default_label_index,
            key="model_label"
        )

        lowercase = st.checkbox("Convert to lowercase", value=True, key="model_lower")
        punctuation = st.checkbox("Remove punctuation", value=True, key="model_punc")
        numbers = st.checkbox("Remove numbers", value=True, key="model_nums")

        st.info("Recommended selections: text column = `post_text`, label column = `label`.")

        if st.button("Train and Evaluate Model"):
            valid, message = validate_model_inputs(df, text_column, label_column)

            if not valid:
                st.error(message)
            else:
                try:
                    df["clean_text"] = clean_text_series(
                        df[text_column],
                        lowercase=lowercase,
                        punctuation=punctuation,
                        numbers=numbers
                    )

                    X = df["clean_text"]
                    y = df[label_column]
                    positive_class = get_positive_class(y)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=15,
                        stratify=y
                    )

                    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
                    X_train_tfidf = vectorizer.fit_transform(X_train)
                    X_test_tfidf = vectorizer.transform(X_test)

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train_tfidf, y_train)

                    y_pred = model.predict(X_test_tfidf)
                    y_prob = get_positive_probability(model, X_test_tfidf, positive_class)

                    y_test_binary = binary_y(y_test, positive_class)

                    fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob)
                    roc_auc_value = auc(fpr, tpr)

                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)

                    advanced_metrics = calculate_advanced_metrics(
                        y_test,
                        y_pred,
                        y_prob,
                        positive_class
                    )

                    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                        y_test_binary,
                        y_prob
                    )
                    average_precision = average_precision_score(y_test_binary, y_prob)

                    threshold_results = evaluate_thresholds(y_test, y_prob, positive_class)
                    cv_scores = run_cross_validation(X, y, positive_class)

                    baseline_results = compare_baseline_models(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        positive_class
                    )

                    refinement_results = compare_refinements(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        positive_class
                    )

                    top_positive, top_negative = plot_feature_importance(
                        model,
                        vectorizer,
                        top_n=20
                    )

                    st.session_state["accuracy"] = accuracy
                    st.session_state["report"] = report
                    st.session_state["cm"] = cm
                    st.session_state["roc_auc_score"] = roc_auc_value
                    st.session_state["advanced_metrics"] = advanced_metrics
                    st.session_state["cv_scores"] = cv_scores
                    st.session_state["threshold_results"] = threshold_results
                    st.session_state["baseline_results"] = baseline_results
                    st.session_state["refinement_results"] = refinement_results

                    st.success("Model trained successfully.")

                    st.subheader("Accuracy")
                    st.write(round(accuracy, 3))

                    st.subheader("Classification Report")
                    st.text(report)

                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(ax=ax)
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                    st.session_state["confusion_matrix_fig"] = fig

                    st.download_button(
                        label="Download Confusion Matrix Plot as PNG",
                        data=fig_to_png_bytes(fig),
                        file_name="confusion_matrix_plot.png",
                        mime="image/png"
                    )

                    st.subheader("ROC-AUC Score")
                    st.write(round(roc_auc_value, 3))

                    st.subheader("ROC Curve")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    roc_display = RocCurveDisplay(
                        fpr=fpr,
                        tpr=tpr,
                        roc_auc=roc_auc_value
                    )
                    roc_display.plot(ax=ax)
                    ax.set_title("ROC Curve")
                    st.pyplot(fig)
                    st.session_state["roc_curve_fig"] = fig

                    st.download_button(
                        label="Download ROC Curve Plot as PNG",
                        data=fig_to_png_bytes(fig),
                        file_name="roc_curve_plot.png",
                        mime="image/png"
                    )

                    st.subheader("Precision-Recall Curve")
                    st.write(f"Average Precision Score: {average_precision:.3f}")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.plot(recall_curve, precision_curve)
                    ax.set_title("Precision-Recall Curve")
                    ax.set_xlabel("Recall")
                    ax.set_ylabel("Precision")
                    st.pyplot(fig)
                    st.session_state["precision_recall_fig"] = fig

                    st.subheader("Sensitivity and Specificity")
                    st.dataframe(pd.DataFrame([advanced_metrics]))

                    st.subheader("Cross-Validation Results")
                    st.write("5-Fold F1 Scores:")
                    st.dataframe(pd.DataFrame(cv_scores, columns=["value"]))
                    st.write("Mean F1 Score:", round(cv_scores.mean(), 3))
                    st.write("Standard Deviation:", round(cv_scores.std(), 3))

                    st.subheader("Threshold Analysis")
                    st.dataframe(threshold_results)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(threshold_results["Threshold"], threshold_results["Recall"], marker="o", label="Recall")
                    ax.plot(threshold_results["Threshold"], threshold_results["Precision"], marker="o", label="Precision")
                    ax.set_title("Threshold Impact on Precision and Recall")
                    ax.set_xlabel("Classification Threshold")
                    ax.set_ylabel("Score")
                    ax.legend()
                    st.pyplot(fig)
                    st.session_state["threshold_fig"] = fig

                    st.subheader("Calibration Curve")
                    prob_true, prob_pred = calibration_curve(
                        y_test_binary,
                        y_prob,
                        n_bins=10
                    )

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.plot(prob_pred, prob_true, marker="o")
                    ax.plot([0, 1], [0, 1], linestyle="--")
                    ax.set_title("Calibration Curve")
                    ax.set_xlabel("Mean Predicted Probability")
                    ax.set_ylabel("Fraction of Positives")
                    st.pyplot(fig)
                    st.session_state["calibration_fig"] = fig

                    st.subheader("Baseline Model Comparison")
                    st.dataframe(baseline_results)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    baseline_results.plot(kind="bar", x="Model", y="F1 Score", legend=False, ax=ax)
                    ax.set_title("Baseline Model Comparison by F1 Score")
                    ax.set_xlabel("Model")
                    ax.set_ylabel("F1 Score")
                    st.pyplot(fig)
                    st.session_state["baseline_fig"] = fig

                    st.subheader("Iterative Refinement Comparison")
                    st.dataframe(refinement_results)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    refinement_results.plot(kind="bar", x="Iteration", y="F1 Score", legend=False, ax=ax)
                    ax.set_title("Iterative Refinement Comparison by F1 Score")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("F1 Score")
                    plt.xticks(rotation=30, ha="right")
                    st.pyplot(fig)
                    st.session_state["refinement_fig"] = fig

                    st.subheader("Feature Importance: Depression-Related Language Indicators")
                    st.dataframe(top_positive)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    top_positive.sort_values("Coefficient").plot(
                        kind="barh",
                        x="Feature",
                        y="Coefficient",
                        legend=False,
                        ax=ax
                    )
                    ax.set_title("Top Positive Features")
                    ax.set_xlabel("Coefficient")
                    st.pyplot(fig)
                    st.session_state["positive_feature_fig"] = fig

                    st.subheader("Feature Importance: Non-Depression-Related Language Indicators")
                    st.dataframe(top_negative)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    top_negative.sort_values("Coefficient", ascending=False).plot(
                        kind="barh",
                        x="Feature",
                        y="Coefficient",
                        legend=False,
                        ax=ax
                    )
                    ax.set_title("Top Negative Features")
                    ax.set_xlabel("Coefficient")
                    st.pyplot(fig)
                    st.session_state["negative_feature_fig"] = fig

                except Exception as e:
                    st.error(f"Model error: {e}")
    else:
        st.info("Please load a dataset first.")


elif section == "Predict New Post":
    st.header("Predict New Social Media Post")

    st.write(
        "Enter one social media post below. This feature uses a separate post-level model "
        "trained on a severity-labeled Reddit dataset to estimate whether the text contains "
        "depression-related language."
    )

    st.caption(
        "This feature is for educational analysis only and should not be used as a clinical diagnosis."
    )

    if "single_post_model" not in st.session_state or "single_post_vectorizer" not in st.session_state:
        try:
            with st.spinner("Loading single-post prediction model..."):
                single_post_model, single_post_vectorizer = train_single_post_model()
                st.session_state["single_post_model"] = single_post_model
                st.session_state["single_post_vectorizer"] = single_post_vectorizer
            st.success("Single-post prediction model loaded successfully.")
        except Exception as e:
            st.error(f"Could not load the single-post prediction model: {e}")

    user_text = st.text_area("Enter a social media post:")

    if st.button("Analyze Post"):
        if user_text.strip() == "":
            st.error("Please enter text before analyzing.")
        elif "single_post_model" not in st.session_state or "single_post_vectorizer" not in st.session_state:
            st.error("The single-post prediction model is not available.")
        else:
            cleaned_text = clean_text_series(pd.Series([user_text]))
            text_tfidf = st.session_state["single_post_vectorizer"].transform(cleaned_text)

            probability = st.session_state["single_post_model"].predict_proba(text_tfidf)[0]

            prob_non_dep = probability[0]
            prob_dep = probability[1]

            st.subheader("Prediction Result")

            if prob_dep > prob_non_dep:
                st.error("Model prediction: depression-related language.")
            else:
                st.success("Model prediction: non-depression-related language.")

            st.write(f"Probability of non-depression-related language: {prob_non_dep:.2%}")
            st.write(f"Probability of depression-related language: {prob_dep:.2%}")

            st.caption(
                "This result is based on learned language patterns from the training dataset. "
                "It should be interpreted cautiously and is not a mental health diagnosis."
            )


elif section == "Generate Report":
    st.header("Generate Report")

    if "accuracy" in st.session_state:
        report_text = generate_report(
            round(st.session_state["accuracy"], 3),
            st.session_state["report"],
            st.session_state["cm"],
            round(st.session_state.get("roc_auc_score", 0), 3),
            st.session_state.get("advanced_metrics"),
            st.session_state.get("cv_scores"),
            st.session_state.get("threshold_results"),
            st.session_state.get("baseline_results"),
            st.session_state.get("refinement_results")
        )

        st.text_area("Generated Report", report_text, height=400)

        st.download_button(
            label="Download Report as TXT File",
            data=report_text,
            file_name="advanced_depression_data_product_report.txt",
            mime="text/plain"
        )

        st.subheader("Download Available Plots")

        st.download_button(
            label="Download All Available Plots as ZIP",
            data=create_plots_zip(),
            file_name="advanced_depression_data_product_plots.zip",
            mime="application/zip"
        )

    else:
        st.warning("Please run the model first before generating a report.")


elif section == "Help":
    st.header("Help")

    st.subheader("How to Use This App")
    st.write("""
1. Open the app using Streamlit.
2. The app will automatically load a default CSV dataset if one is available.
3. If needed, upload a CSV file using the sidebar.
4. Use Explore Data to view the dataset, missing values, class distribution, text length, and average text length by class.
5. Use Preprocess Text to clean the text data.
6. Use Run Model to train and evaluate the Logistic Regression model.
7. Review advanced evaluation outputs including ROC-AUC, Precision-Recall Curve, Cross-Validation, Sensitivity, Specificity, Calibration Curve, Threshold Analysis, Baseline Model Comparison, Iterative Refinement, and Feature Importance.
8. Use Predict New Post to enter a single social media post and view prediction probabilities.
9. Use Generate Report to create and download the model report and available plots.
""")

    st.subheader("Required Dataset Columns")
    st.write("""
Recommended columns for the main dataset:
- `post_text`: contains the social media text.
- `label`: contains the classification target, usually 0 or 1.
""")

    st.subheader("Advanced Analysis Features")
    st.write("""
- ROC-AUC evaluates how well the model separates classes across thresholds.
- Precision-Recall Curve shows the tradeoff between identifying depression-related posts and limiting false positives.
- Sensitivity measures how well the model identifies depression-related posts.
- Specificity measures how well the model identifies non-depression-related posts.
- Cross-validation evaluates model stability across multiple folds.
- Threshold analysis shows how changing the decision cutoff affects false positives and false negatives.
- Calibration analysis evaluates whether predicted probabilities are reliable.
- Baseline comparison compares Logistic Regression against Naive Bayes, Linear SVM, and Random Forest.
- Iterative refinement compares the original TF-IDF setup against expanded feature settings.
- Feature importance identifies the terms most associated with each prediction class.
""")

    st.subheader("Security and Privacy")
    st.write("""
This app does not require personal identifiers, usernames, passwords, or private information.
The data is processed only while the app is running. User-entered text is analyzed only during
the active app session. The product is intended for educational analysis and is not a replacement
for clinical diagnosis or professional mental health care.
""")
