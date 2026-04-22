import numpy as np
import pytest
from app.features import extract_query_features, FEATURE_NAMES, generate_training_corpus


def test_feature_vector_shape():
    features = extract_query_features("Hello world, how are you?")
    assert features.shape == (len(FEATURE_NAMES),)


def test_feature_dtype():
    features = extract_query_features("Test query")
    assert features.dtype == np.float32


def test_char_len_feature():
    q = "Hello world"
    features = extract_query_features(q)
    assert features[0] == len(q)


def test_word_count_feature():
    q = "one two three four"
    features = extract_query_features(q)
    assert features[1] == 4


def test_lexical_diversity_range():
    q = "the the the the the"
    features = extract_query_features(q)
    diversity = features[2]
    assert 0.0 <= diversity <= 1.0


def test_sql_keyword_detection():
    normal = extract_query_features("What is supervised learning?")
    sql_query = extract_query_features("SELECT * FROM users WHERE id=1 UNION SELECT password")
    assert sql_query[13] > normal[13]


def test_lag_features_with_history():
    history = ["short", "a somewhat longer query than before"]
    q = "What is machine learning?"
    features = extract_query_features(q, history)
    assert features[8] != 1.0 or features[9] != 1.0


def test_rolling_features_empty_history():
    features = extract_query_features("No history query")
    rolling_mean = features[10]
    rolling_std = features[11]
    assert rolling_std == 0.0
    assert rolling_mean > 0


def test_generate_training_corpus_shape():
    X, y = generate_training_corpus(n_normal=50, n_anomaly=10, seed=1)
    assert X.shape == (60, len(FEATURE_NAMES))
    assert y.shape == (60,)
    assert y.sum() == 10


def test_generate_training_corpus_labels():
    X, y = generate_training_corpus(n_normal=20, n_anomaly=5, seed=2)
    assert set(y.tolist()) == {0, 1}


def test_anomalous_query_has_sql_features():
    features = extract_query_features("'; DROP TABLE users; INSERT INTO logs VALUES(1)")
    assert features[13] > 0  # sql_keywords


def test_special_char_ratio_for_injection():
    normal = extract_query_features("What is deep learning?")
    injected = extract_query_features("SELECT {exec} (DROP) <script> [alert]")
    assert injected[7] > normal[7]
