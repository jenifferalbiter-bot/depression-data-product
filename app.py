import re
from pathlib import Path
from io import BytesIO
import zipfile

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay
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
        }

        for filename, fig in plot_files.items():
            if fig is not None:
                img_buffer = fig_to_png_bytes(fig)
                zip_file.writestr(filename, img_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer


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


def generate_report(accuracy, report_text, cm, roc_auc_score=None):
    return f"""
Depression-Related Language Detection Data Product Report

Purpose:
This report summarizes the results of the NLP data product. The product uses text preprocessing,
TF-IDF vectorization, Logistic Regression, and single-post prediction to identify depression-related
language patterns in social media text.

Data Analysis Methods:
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification
- Single-post prediction using a separate post-level model
- Probability scoring for single-post predictions
- Accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC evaluation
- Exportable plots in PNG format
- Combined plot export in ZIP format

Accuracy:
{accuracy}

ROC-AUC:
{roc_auc_score}

Classification Report:
{report_text}

Confusion Matrix:
{cm}

Exportable Reports and Plots:
The application allows users to download the model report as a TXT file. It also allows users
to export generated plots, including the class distribution plot, text length distribution plot,
confusion matrix plot, and ROC curve plot, as PNG image files. Available plots can also be
downloaded together as a ZIP file.

Single-Post Prediction:
The application includes a Predict New Post feature that allows users to enter one social media
post and receive a model prediction. This feature displays probability scores for both classes:
non-depression-related language and depression-related language. These probability scores help
users interpret model confidence in addition to the predicted class.

Security and Privacy:
- The app does not require names, usernames, passwords, or personal identifiers.
- The dataset is processed only during the active app session.
- User-entered text is analyzed only during the active app session.
- The app is intended for educational and research analysis only.
- The app is not a clinical diagnosis tool.

Help:
Users can upload a CSV file or use the default dataset. The required columns are a text column
such as post_text and a label column such as label. Users may also enter one social media post
in the Predict New Post section to view a model prediction and probability scores.
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

                    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc_score = auc(fpr, tpr)

                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)

                    st.session_state["accuracy"] = accuracy
                    st.session_state["report"] = report
                    st.session_state["cm"] = cm
                    st.session_state["roc_auc_score"] = roc_auc_score
                    st.session_state["fpr"] = fpr
                    st.session_state["tpr"] = tpr

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
                    st.write(round(roc_auc_score, 3))

                    st.subheader("ROC Curve")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    roc_display = RocCurveDisplay(
                        fpr=fpr,
                        tpr=tpr,
                        roc_auc=roc_auc_score
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
            round(st.session_state.get("roc_auc_score", 0), 3)
        )

        st.text_area("Generated Report", report_text, height=400)

        st.download_button(
            label="Download Report as TXT File",
            data=report_text,
            file_name="depression_data_product_report.txt",
            mime="text/plain"
        )

        st.subheader("Download Available Plots")

        if (
            "class_distribution_fig" in st.session_state
            or "text_length_fig" in st.session_state
            or "confusion_matrix_fig" in st.session_state
            or "roc_curve_fig" in st.session_state
        ):
            st.download_button(
                label="Download All Available Plots as ZIP",
                data=create_plots_zip(),
                file_name="depression_data_product_plots.zip",
                mime="application/zip"
            )

            if "class_distribution_fig" in st.session_state:
                st.download_button(
                    label="Download Class Distribution Plot as PNG",
                    data=fig_to_png_bytes(st.session_state["class_distribution_fig"]),
                    file_name="class_distribution_plot.png",
                    mime="image/png"
                )

            if "text_length_fig" in st.session_state:
                st.download_button(
                    label="Download Text Length Distribution Plot as PNG",
                    data=fig_to_png_bytes(st.session_state["text_length_fig"]),
                    file_name="text_length_distribution_plot.png",
                    mime="image/png"
                )

            if "confusion_matrix_fig" in st.session_state:
                st.download_button(
                    label="Download Confusion Matrix Plot as PNG",
                    data=fig_to_png_bytes(st.session_state["confusion_matrix_fig"]),
                    file_name="confusion_matrix_plot.png",
                    mime="image/png"
                )

            if "roc_curve_fig" in st.session_state:
                st.download_button(
                    label="Download ROC Curve Plot as PNG",
                    data=fig_to_png_bytes(st.session_state["roc_curve_fig"]),
                    file_name="roc_curve_plot.png",
                    mime="image/png"
                )
        else:
            st.info("Explore the data or run the model first to generate downloadable plots.")

    else:
        st.warning("Please run the model first before generating a report.")


elif section == "Help":
    st.header("Help")

    st.subheader("How to Use This App")
    st.write("""
1. Open the app using Streamlit.
2. The app will automatically load a default CSV dataset if one is available.
3. If needed, upload a CSV file using the sidebar.
4. Use Explore Data to view the dataset, missing values, class distribution, and text length.
5. Use Preprocess Text to clean the text data.
6. Use Run Model to train and evaluate the Logistic Regression model.
7. Use Predict New Post to enter a single social media post and view the prediction probabilities.
8. Use Generate Report to create and download the model report.
9. Download individual plots as PNG files or download all available plots together as a ZIP file.
""")

    st.subheader("Required Dataset Columns")
    st.write("""
Recommended columns for the main dataset:
- `post_text`: contains the social media text.
- `label`: contains the classification target, usually 0 or 1.
""")

    st.subheader("Function Descriptions")
    st.write("""
- Upload Data: Loads the default or uploaded CSV dataset.
- Explore Data: Displays dataset structure, missing values, class distribution, and text length visualizations.
- Preprocess Text: Cleans text by lowercasing and removing punctuation or numbers.
- Run Model: Uses TF-IDF and Logistic Regression to classify text from the uploaded/default dataset.
- Predict New Post: Uses a separate post-level model to classify one user-entered social media post.
- Generate Report: Saves model results to a downloadable text file and allows generated plots to be exported.
- Help: Explains how to use the product.
""")

    st.subheader("Report and Plot Export")
    st.write("""
The app supports exporting the model report as a TXT file. It also supports exporting generated
visualizations as PNG image files. Available plots include the class distribution plot, text length
distribution plot, confusion matrix plot, and ROC curve plot. Users can also download all available
plots together as a ZIP file.
""")

    st.subheader("Security and Privacy")
    st.write("""
This app does not require personal identifiers, usernames, passwords, or private information.
The data is processed only while the app is running. User-entered text is analyzed only during
the active app session. The product is intended for educational analysis and is not a replacement
for clinical diagnosis or professional mental health care.
""")
