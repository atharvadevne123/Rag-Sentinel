import numpy as np

from app.features import FEATURE_NAMES, extract_query_features, generate_training_corpus


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


import pytest


@pytest.mark.parametrize("query,expected_idx,expected_nonzero", [
    ("SELECT * FROM users", 13, True),
    ("DROP TABLE admin", 13, True),
    ("What is gradient descent?", 13, False),
    ("UNION ALL SELECT password", 13, True),
    ("Explain BERT embeddings", 13, False),
])
def test_sql_keyword_feature_parametrized(query, expected_idx, expected_nonzero):
    features = extract_query_features(query)
    if expected_nonzero:
        assert features[expected_idx] > 0
    else:
        assert features[expected_idx] == 0


@pytest.mark.parametrize("length", [1, 10, 100, 500, 2000])
def test_char_len_feature_various_lengths(length):
    query = "a" * length
    features = extract_query_features(query)
    assert features[0] == length


def test_generate_corpus_balance():
    X, y = generate_training_corpus(n_normal=60, n_anomaly=20, seed=7)
    assert int(y.sum()) == 20
    assert int((y == 0).sum()) == 60


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 15


def test_extract_features_with_long_history():
    history = ["query " + str(i) for i in range(20)]
    features = extract_query_features("test query", history)
    assert features.shape == (15,)


def test_punct_ratio_increases_with_punctuation():
    few_punct = extract_query_features("Hello world how are you today")
    many_punct = extract_query_features("Hello! World? How, are; you: today.")
    assert many_punct[4] > few_punct[4]


def test_upper_ratio_increases_with_caps():
    lower = extract_query_features("what is machine learning")
    upper = extract_query_features("WHAT IS MACHINE LEARNING")
    assert upper[6] > lower[6]


def test_digit_ratio_increases_with_digits():
    no_digits = extract_query_features("what is machine learning")
    with_digits = extract_query_features("123 456 789 what is 42 learning")
    assert with_digits[5] > no_digits[5]
