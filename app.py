from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
except ImportError:
    ENGLISH_STOP_WORDS = set()
    TfidfVectorizer = None
    accuracy_score = None
    train_test_split = None
    MultinomialNB = None

try:
    import nltk
    from nltk.corpus import stopwords
except ImportError:
    nltk = None
    stopwords = None


APP_TITLE = "\U0001F4E7 Spam Email Detection using NLP"
PAGE_TITLE = "Spam Email Detection using NLP"
MODEL_NAME = "Optimized TF-IDF (1-3 grams) + Multinomial Naive Bayes"
DATASET_NAME = "spam_ham_dataset.csv"
INPUT_KEY = "email_input"
RESULT_KEY = "prediction_payload"
SELF_CHECK_KEY = "self_check_results"
NB_ALPHA = 0.3
TFIDF_CONFIG = {
    "ngram_range": (1, 3),
    "min_df": 2,
    "sublinear_tf": True,
}
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
EXAMPLE_EMAILS = [
    (
        "Normal Email Example",
        "Use Normal Email",
        "Hi John, just checking if we are still meeting tomorrow at 10 AM. Let me know.",
    ),
    (
        "Spam Email Example",
        "Use Spam Email",
        "Congratulations! You have won a $1000 Amazon gift card. Click here to claim now!",
    ),
    (
        "Phishing Email Example",
        "Use Phishing Email",
        "URGENT: Your bank account has been compromised. Login immediately to verify your identity at http://secure-update-login.com",
    ),
]
SELF_CHECK_CASES = [
    {
        "name": "Team Follow-up",
        "expected": 0,
        "text": "Hi Sarah, can you please share the updated slides before today's 4 PM review meeting?",
    },
    {
        "name": "Family Message",
        "expected": 0,
        "text": "Mom, I reached safely. I will call you after dinner tonight.",
    },
    {
        "name": "Prize Scam",
        "expected": 1,
        "text": "Congratulations! You have been selected to win a free iPhone. Click now to claim your reward.",
    },
    {
        "name": "Bank Phishing",
        "expected": 1,
        "text": "URGENT: Verify your bank account immediately or your online access will be suspended. Click the secure link now.",
    },
]


@dataclass
class ModelArtifacts:
    model: MultinomialNB
    vectorizer: TfidfVectorizer
    accuracy: float
    stop_words: set[str]
    row_count: int
    ham_class_index: int
    spam_class_index: int


def ensure_dependencies() -> None:
    missing_packages: list[str] = []

    if pd is None:
        missing_packages.append("pandas")
    if TfidfVectorizer is None or MultinomialNB is None:
        missing_packages.append("scikit-learn")
    if nltk is None or stopwords is None:
        missing_packages.append("nltk")

    if not missing_packages:
        return

    st.error(
        "Missing required Python packages: "
        + ", ".join(missing_packages)
        + ". Install them before running the app."
    )
    st.code("pip install pandas scikit-learn nltk streamlit", language="bash")
    st.stop()


def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 48%, #f8fafc 100%);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 1.5rem;
        }
        .app-title {
            text-align: center;
            font-size: 2.7rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }
        .app-subtitle {
            text-align: center;
            color: #475569;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        .info-label {
            color: #64748b;
            font-size: 0.92rem;
            margin-bottom: 0.25rem;
        }
        .info-value {
            color: #0f172a;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.35;
            margin-bottom: 0.8rem;
        }
        .footer {
            text-align: center;
            color: #64748b;
            font-size: 0.95rem;
            padding-top: 1rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 0.65rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid #cbd5e1;
            font-weight: 600;
            min-height: 2.8rem;
        }
        textarea {
            border-radius: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    st.session_state.setdefault(INPUT_KEY, "")
    st.session_state.setdefault(RESULT_KEY, None)
    st.session_state.setdefault(SELF_CHECK_KEY, None)


def clear_prediction() -> None:
    st.session_state[RESULT_KEY] = None


def load_example(example_text: str) -> None:
    st.session_state[INPUT_KEY] = example_text
    clear_prediction()


def candidate_dataset_paths() -> list[Path]:
    home = Path.home()
    exact_user_path = Path(
        r"C:\Users\mahar\OneDrive\Desktop\NLP FT-3\archive\spam_ham_dataset.csv"
    )
    env_path = os.getenv("SPAM_DATASET_PATH")

    candidates = [
        Path(__file__).resolve().parent / DATASET_NAME,
        Path.cwd() / DATASET_NAME,
        home / DATASET_NAME,
        home / "Desktop" / DATASET_NAME,
        home / "Desktop" / "NLP FT-3" / "archive" / DATASET_NAME,
        home / "OneDrive" / "Desktop" / DATASET_NAME,
        home / "OneDrive" / "Desktop" / "NLP FT-3" / "archive" / DATASET_NAME,
        exact_user_path,
    ]

    if env_path:
        candidates.insert(0, Path(env_path).expanduser())

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate_key)

    return unique_candidates


def find_dataset_path() -> Path | None:
    for candidate in candidate_dataset_paths():
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    normalized_columns = {
        str(column).strip().lower(): column for column in dataframe.columns
    }

    label_column = normalized_columns.get("label")
    text_column = normalized_columns.get("text") or normalized_columns.get("message")

    if label_column is None or text_column is None:
        available = ", ".join(map(str, dataframe.columns))
        raise ValueError(
            "Required columns are missing. Expected 'label' and either 'text' or "
            f"'message'. Available columns: {available}"
        )

    return label_column, text_column


def get_stop_words() -> tuple[set[str], str]:
    fallback_stop_words = set(ENGLISH_STOP_WORDS) if ENGLISH_STOP_WORDS else set()

    if nltk is None or stopwords is None:
        return fallback_stop_words, "sklearn fallback"

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            return fallback_stop_words, "sklearn fallback"

    try:
        return set(stopwords.words("english")), "nltk"
    except LookupError:
        return fallback_stop_words, "sklearn fallback"


def normalize_label(value: object) -> int | None:
    if value is None or (pd is not None and pd.isna(value)):
        return None

    normalized = str(value).strip().lower()
    if normalized in {"spam", "1", "true", "yes"}:
        return 1
    if normalized in {"ham", "0", "false", "no"}:
        return 0
    return None


def clean_text(text: object, stop_words: set[str]) -> str:
    if text is None or (pd is not None and pd.isna(text)):
        return ""

    lowered_text = str(text).lower()
    no_special_chars = SPECIAL_CHAR_PATTERN.sub(" ", lowered_text)
    normalized_text = WHITESPACE_PATTERN.sub(" ", no_special_chars).strip()
    tokens = normalized_text.split()
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)


def prepare_training_data(
    dataframe: pd.DataFrame,
    label_column: str,
    text_column: str,
    stop_words: set[str],
) -> tuple[pd.Series, pd.Series]:
    working_frame = dataframe[[label_column, text_column]].copy()
    working_frame[text_column] = working_frame[text_column].fillna("").astype(str)

    raw_labels = working_frame[label_column].copy()
    working_frame["encoded_label"] = raw_labels.apply(normalize_label)

    invalid_labels = working_frame["encoded_label"].isna()
    if invalid_labels.any():
        unexpected_values = sorted(raw_labels[invalid_labels].astype(str).unique())
        raise ValueError(
            "Unsupported label values found in the dataset: "
            + ", ".join(unexpected_values)
        )

    working_frame["clean_text"] = working_frame[text_column].apply(
        lambda value: clean_text(value, stop_words)
    )
    working_frame = working_frame[working_frame["clean_text"].str.strip().ne("")]

    if working_frame.empty:
        raise ValueError("No usable text rows remain after preprocessing.")

    labels = working_frame["encoded_label"].astype(int)
    texts = working_frame["clean_text"]

    if labels.nunique() < 2:
        raise ValueError("The dataset must contain both spam and ham labels.")

    class_counts = labels.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Each class must contain at least two samples for training and evaluation."
        )

    return texts, labels


def compute_test_size(labels: pd.Series) -> int:
    class_count = int(labels.nunique())
    suggested_size = max(class_count, round(len(labels) * 0.2))
    max_allowed_size = len(labels) - class_count

    if max_allowed_size < class_count:
        raise ValueError("The dataset is too small to split into train and test sets.")

    return min(suggested_size, max_allowed_size)


@st.cache_resource(show_spinner="Training the spam detection model...")
def train_model(dataset_path: str) -> ModelArtifacts:
    dataframe = pd.read_csv(dataset_path)
    label_column, text_column = resolve_columns(dataframe)
    stop_words, _ = get_stop_words()
    texts, labels = prepare_training_data(dataframe, label_column, text_column, stop_words)

    test_size = compute_test_size(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)

    model = MultinomialNB(alpha=NB_ALPHA)
    model.fit(x_train_features, y_train)

    predictions = model.predict(x_test_features)
    accuracy = accuracy_score(y_test, predictions)
    ham_class_index = int(list(model.classes_).index(0))
    spam_class_index = int(list(model.classes_).index(1))

    return ModelArtifacts(
        model=model,
        vectorizer=vectorizer,
        accuracy=float(accuracy),
        stop_words=stop_words,
        row_count=int(len(texts)),
        ham_class_index=ham_class_index,
        spam_class_index=spam_class_index,
    )


def predict_email(artifacts: ModelArtifacts, email_text: str) -> dict[str, float | int | str]:
    cleaned_input = clean_text(email_text, artifacts.stop_words)
    if not cleaned_input:
        raise ValueError(
            "The input became empty after preprocessing. Please enter more meaningful email content."
        )

    feature_vector = artifacts.vectorizer.transform([cleaned_input])
    prediction = int(artifacts.model.predict(feature_vector)[0])
    probabilities = artifacts.model.predict_proba(feature_vector)[0]
    predicted_class_index = (
        artifacts.spam_class_index if prediction == 1 else artifacts.ham_class_index
    )
    confidence = float(probabilities[predicted_class_index])

    return {
        "prediction": prediction,
        "label": "Spam" if prediction == 1 else "Not Spam",
        "confidence": confidence,
        "spam_probability": float(probabilities[artifacts.spam_class_index]),
        "ham_probability": float(probabilities[artifacts.ham_class_index]),
    }


def run_self_check(artifacts: ModelArtifacts) -> list[dict[str, float | int | str | bool]]:
    results: list[dict[str, float | int | str | bool]] = []

    for case in SELF_CHECK_CASES:
        prediction_result = predict_email(artifacts, case["text"])
        expected_label = "Spam" if case["expected"] == 1 else "Not Spam"
        results.append(
            {
                "name": case["name"],
                "text": case["text"],
                "expected": case["expected"],
                "expected_label": expected_label,
                "predicted_label": prediction_result["label"],
                "confidence": prediction_result["confidence"],
                "passed": prediction_result["prediction"] == case["expected"],
            }
        )

    return results


def render_title() -> None:
    st.markdown(f"<div class='app-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='app-subtitle'>Paste an email below or try an example to classify it with NLP and machine learning.</div>",
        unsafe_allow_html=True,
    )


def render_model_info(artifacts: ModelArtifacts) -> None:
    info_col, metric_col = st.columns([2.2, 1])

    with info_col:
        with st.container(border=True):
            st.subheader("Model Info")
            details_col_1, details_col_2 = st.columns(2)
            with details_col_1:
                st.markdown("<div class='info-label'>Dataset size</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='info-value'>{artifacts.row_count:,} emails</div>",
                    unsafe_allow_html=True,
                )
            with details_col_2:
                st.markdown("<div class='info-label'>Model used</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='info-value'>{MODEL_NAME}</div>",
                    unsafe_allow_html=True,
                )

    with metric_col:
        with st.container(border=True):
            st.metric("Model Accuracy", f"{artifacts.accuracy:.2%}")
            st.caption("Measured on held-out test data")


def render_example_section() -> None:
    with st.container(border=True):
        st.subheader("Try Example Emails")
        example_columns = st.columns(3)

        for column, (title, button_label, example_text) in zip(
            example_columns,
            EXAMPLE_EMAILS,
        ):
            with column:
                st.markdown(f"**{title}**")
                st.write(example_text)
                if st.button(button_label, key=button_label, use_container_width=True):
                    load_example(example_text)


def render_self_check_section(artifacts: ModelArtifacts) -> None:
    with st.container(border=True):
        header_col, action_col = st.columns([2.4, 1])

        with header_col:
            st.subheader("Quick Self-Check")
            st.caption(
                "Run a few built-in examples to verify how the model behaves on common safe and risky email patterns."
            )

        with action_col:
            st.write("")
            if st.button("Run Self-Check", key="run_self_check", use_container_width=True):
                with st.spinner("Running built-in checks..."):
                    st.session_state[SELF_CHECK_KEY] = run_self_check(artifacts)

        results = st.session_state.get(SELF_CHECK_KEY)
        if not results:
            st.info("Click Run Self-Check to test the model on a few built-in emails.")
            return

        passed_count = sum(1 for result in results if result["passed"])
        st.metric("Checks Passed", f"{passed_count}/{len(results)}")

        result_columns = st.columns(2)
        for index, result in enumerate(results):
            with result_columns[index % 2]:
                with st.container(border=True):
                    if result["passed"]:
                        st.success(f"{result['name']}: Passed")
                    else:
                        st.error(f"{result['name']}: Check this prediction")
                    st.caption(
                        f"Expected: {result['expected_label']} | "
                        f"Predicted: {result['predicted_label']} | "
                        f"Confidence: {result['confidence']:.2%}"
                    )
                    st.write(result["text"])


def render_input_section() -> None:
    with st.container(border=True):
        st.subheader("Enter Your Email")
        st.text_area(
            "Email content",
            key=INPUT_KEY,
            height=240,
            placeholder="Paste or type the email content here, then click Predict.",
            label_visibility="collapsed",
            on_change=clear_prediction,
        )


def render_prediction_section() -> None:
    with st.container(border=True):
        st.subheader("Prediction Result")
        result = st.session_state.get(RESULT_KEY)

        if not result:
            st.info("Prediction results will appear here after you click Predict.")
            return

        if result["prediction"] == 1:
            st.error("Result: Spam")
        else:
            st.success("Result: Not Spam")

        st.metric("Confidence Score", f"{result['confidence']:.2%}")
        st.progress(result["confidence"])
        st.caption(
            f"Spam probability: {result['spam_probability']:.2%} | "
            f"Not spam probability: {result['ham_probability']:.2%}"
        )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="\U0001F4E7", layout="wide")

    ensure_dependencies()
    apply_custom_styles()
    initialize_session_state()

    dataset_path = find_dataset_path()
    if dataset_path is None:
        st.error(
            "Dataset not found. Place 'spam_ham_dataset.csv' in the app folder, run "
            "the app from the dataset folder, or set the SPAM_DATASET_PATH "
            "environment variable."
        )
        st.stop()

    try:
        artifacts = train_model(str(dataset_path))
    except FileNotFoundError:
        st.error("Dataset not found. Please check the CSV file path and try again.")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Unable to load the dataset or train the model: {exc}")
        st.stop()

    render_title()
    render_model_info(artifacts)
    render_example_section()
    render_self_check_section(artifacts)

    input_col, result_col = st.columns([1.25, 1])

    with input_col:
        render_input_section()

        if st.button("Predict", type="primary", use_container_width=True):
            email_text = st.session_state.get(INPUT_KEY, "")
            if not email_text.strip():
                st.warning("Please enter an email message before clicking Predict.")
            else:
                with st.spinner("Analyzing email content..."):
                    try:
                        st.session_state[RESULT_KEY] = predict_email(artifacts, email_text)
                    except ValueError as exc:
                        clear_prediction()
                        st.warning(str(exc))

    with result_col:
        render_prediction_section()

    st.markdown("<div class='footer'>Built using NLP + Machine Learning</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
