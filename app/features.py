import re
from typing import List, Tuple

import numpy as np


def _compute_rolling_stats(query: str, history: List[str]) -> Tuple[float, float, float, float, float]:
    """Compute rolling and lag statistics relative to recent query history.

    Args:
        query: Current query string.
        history: List of preceding query strings (most-recent last).

    Returns:
        Tuple of (lag1_ratio, lag2_ratio, rolling_mean, rolling_std, len_deviation).
    """
    char_len = len(query)
    lag1_len = len(history[-1]) if len(history) >= 1 else char_len
    lag2_len = len(history[-2]) if len(history) >= 2 else char_len
    lag1_ratio = char_len / max(lag1_len, 1)
    lag2_ratio = char_len / max(lag2_len, 1)

    if history:
        recent_lens = [len(h) for h in history[-5:]]
        rolling_mean = float(np.mean(recent_lens))
        rolling_std = float(np.std(recent_lens)) if len(recent_lens) > 1 else 0.0
    else:
        rolling_mean = float(char_len)
        rolling_std = 0.0

    len_deviation = abs(char_len - rolling_mean) / max(rolling_mean, 1)
    return lag1_ratio, lag2_ratio, rolling_mean, rolling_std, len_deviation


def _compute_injection_features(query: str) -> Tuple[int, int]:
    """Count SQL-keyword and bracket-style code-injection signals.

    Args:
        query: Query string to inspect.

    Returns:
        Tuple of (sql_keyword_count, code_bracket_count).
    """
    sql_keywords = len(re.findall(r"\b(select|drop|insert|union|exec|script)\b", query.lower()))
    code_pattern = len(re.findall(r"[{}()\[\]<>]", query))
    return sql_keywords, code_pattern


def extract_query_features(query: str, history: List[str] = None) -> np.ndarray:
    """Extract a numeric feature vector from a text query for anomaly scoring.

    Args:
        query: Query string to featurise.
        history: List of recent preceding queries (most-recent last). Defaults
            to an empty list.

    Returns:
        Float32 numpy array of length 15 (see FEATURE_NAMES for ordering).
    """
    history = history or []

    char_len = len(query)
    word_count = len(query.split())
    unique_words = len(set(query.lower().split()))
    lexical_diversity = unique_words / max(word_count, 1)
    avg_word_len = float(np.mean([len(w) for w in query.split()])) if query.split() else 0.0

    punct_ratio = sum(1 for c in query if c in "!?.,;:") / max(char_len, 1)
    digit_ratio = sum(1 for c in query if c.isdigit()) / max(char_len, 1)
    upper_ratio = sum(1 for c in query if c.isupper()) / max(char_len, 1)
    special_ratio = sum(1 for c in query if not c.isalnum() and not c.isspace()) / max(char_len, 1)

    lag1_ratio, lag2_ratio, rolling_mean, rolling_std, len_deviation = _compute_rolling_stats(query, history)
    sql_keywords, code_pattern = _compute_injection_features(query)

    return np.array(
        [
            char_len,
            word_count,
            lexical_diversity,
            avg_word_len,
            punct_ratio,
            digit_ratio,
            upper_ratio,
            special_ratio,
            lag1_ratio,
            lag2_ratio,
            rolling_mean,
            rolling_std,
            len_deviation,
            sql_keywords,
            code_pattern,
        ],
        dtype=np.float32,
    )


FEATURE_NAMES = [
    "char_len",
    "word_count",
    "lexical_diversity",
    "avg_word_len",
    "punct_ratio",
    "digit_ratio",
    "upper_ratio",
    "special_ratio",
    "len_lag1_ratio",
    "len_lag2_ratio",
    "rolling_mean_len",
    "rolling_std_len",
    "len_deviation",
    "sql_keywords",
    "code_pattern",
]


def generate_training_corpus(
    n_normal: int = 500,
    n_anomaly: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic labelled dataset for anomaly classifier training.

    Args:
        n_normal: Number of normal (label=0) samples to generate.
        n_anomaly: Number of anomalous (label=1) samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) where X has shape (n_normal+n_anomaly, 15) and
        y is a binary label vector.
    """
    rng = np.random.default_rng(seed)

    normal_queries = [
        "What is machine learning?",
        "Explain neural networks in simple terms.",
        "How does gradient descent work?",
        "What are the key benefits of transfer learning?",
        "Describe attention mechanisms in transformers.",
        "How do convolutional neural networks process images?",
        "What is the difference between supervised and unsupervised learning?",
        "How does BERT handle contextual embeddings?",
        "Explain the bias-variance tradeoff.",
        "What is regularization and why is it important?",
    ]

    anomalous_queries = [
        "DROP TABLE users; SELECT * FROM admin--",
        "A" * 500,
        "<script>alert('xss')</script>",
        "SELECT password FROM users WHERE 1=1 UNION ALL SELECT null",
        "!!!???!!!???!!!???",
        "x" * 300 + " normal words here",
        "EXEC xp_cmdshell('dir c:\\')",
        "'; INSERT INTO logs VALUES(1,1,1)--",
    ]

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    history: List[str] = []

    for _ in range(n_normal):
        q = str(rng.choice(normal_queries))
        noise = " ".join(rng.choice(["the", "a", "how", "why", "when"], size=rng.integers(0, 4)))
        q = q + " " + noise
        X_list.append(extract_query_features(q, history[-5:]))
        y_list.append(0)
        history.append(q)

    for _ in range(n_anomaly):
        q = str(rng.choice(anomalous_queries))
        X_list.append(extract_query_features(q, history[-5:]))
        y_list.append(1)
        history.append(q)

    return np.array(X_list), np.array(y_list)
