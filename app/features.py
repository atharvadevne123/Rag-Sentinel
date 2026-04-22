import re
import numpy as np
from typing import List


def extract_query_features(query: str, history: List[str] = None) -> np.ndarray:
    """Extract numeric feature vector from a text query for anomaly scoring."""
    history = history or []

    char_len = len(query)
    word_count = len(query.split())
    unique_words = len(set(query.lower().split()))
    lexical_diversity = unique_words / max(word_count, 1)
    avg_word_len = np.mean([len(w) for w in query.split()]) if query.split() else 0.0

    punct_ratio = sum(1 for c in query if c in "!?.,;:") / max(char_len, 1)
    digit_ratio = sum(1 for c in query if c.isdigit()) / max(char_len, 1)
    upper_ratio = sum(1 for c in query if c.isupper()) / max(char_len, 1)
    special_ratio = sum(1 for c in query if not c.isalnum() and not c.isspace()) / max(char_len, 1)

    # Lag features from recent query history
    lag1_len = len(history[-1]) if len(history) >= 1 else char_len
    lag2_len = len(history[-2]) if len(history) >= 2 else char_len
    len_lag1_ratio = char_len / max(lag1_len, 1)
    len_lag2_ratio = char_len / max(lag2_len, 1)

    # Rolling window features
    if history:
        recent_lens = [len(h) for h in history[-5:]]
        rolling_mean_len = float(np.mean(recent_lens))
        rolling_std_len = float(np.std(recent_lens)) if len(recent_lens) > 1 else 0.0
    else:
        rolling_mean_len = float(char_len)
        rolling_std_len = 0.0

    # Deviation from rolling mean
    len_deviation = abs(char_len - rolling_mean_len) / max(rolling_mean_len, 1)

    # SQL/code injection heuristic features
    sql_keywords = len(re.findall(r"\b(select|drop|insert|union|exec|script)\b", query.lower()))
    code_pattern = len(re.findall(r"[{}()\[\]<>]", query))

    features = np.array([
        char_len,
        word_count,
        lexical_diversity,
        avg_word_len,
        punct_ratio,
        digit_ratio,
        upper_ratio,
        special_ratio,
        len_lag1_ratio,
        len_lag2_ratio,
        rolling_mean_len,
        rolling_std_len,
        len_deviation,
        sql_keywords,
        code_pattern,
    ], dtype=np.float32)

    return features


FEATURE_NAMES = [
    "char_len", "word_count", "lexical_diversity", "avg_word_len",
    "punct_ratio", "digit_ratio", "upper_ratio", "special_ratio",
    "len_lag1_ratio", "len_lag2_ratio", "rolling_mean_len", "rolling_std_len",
    "len_deviation", "sql_keywords", "code_pattern",
]


def generate_training_corpus(n_normal: int = 500, n_anomaly: int = 50, seed: int = 42) -> tuple:
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

    X_list, y_list = [], []

    history: List[str] = []
    for _ in range(n_normal):
        q = rng.choice(normal_queries)
        noise = " ".join(rng.choice(["the", "a", "how", "why", "when"], size=rng.integers(0, 4)))
        q = q + " " + noise
        feats = extract_query_features(q, history[-5:])
        X_list.append(feats)
        y_list.append(0)
        history.append(q)

    for _ in range(n_anomaly):
        q = rng.choice(anomalous_queries)
        feats = extract_query_features(q, history[-5:])
        X_list.append(feats)
        y_list.append(1)
        history.append(q)

    return np.array(X_list), np.array(y_list)
