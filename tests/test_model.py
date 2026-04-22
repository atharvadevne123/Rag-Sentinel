from app.features import extract_query_features, generate_training_corpus
from app.model import train_model, predict_anomaly


def test_train_model_returns_bundle_and_metrics():
    X, y = generate_training_corpus(n_normal=80, n_anomaly=20, seed=10)
    bundle, metrics = train_model(X, y)
    assert "classifier" in bundle
    assert "isolation_forest" in bundle
    assert "auc_mean" in metrics
    assert "auc_std" in metrics


def test_model_auc_above_threshold():
    X, y = generate_training_corpus(n_normal=100, n_anomaly=25, seed=11)
    _, metrics = train_model(X, y)
    assert metrics["auc_mean"] > 0.6, f"AUC too low: {metrics['auc_mean']}"


def test_predict_normal_query(trained_model):
    bundle, _ = trained_model
    features = extract_query_features("What is machine learning?", [])
    result = predict_anomaly(bundle, features)
    assert "is_anomaly" in result
    assert "anomaly_score" in result
    assert 0.0 <= result["anomaly_score"] <= 1.0


def test_predict_sql_injection_flagged(trained_model):
    bundle, _ = trained_model
    sql = "DROP TABLE users; SELECT * FROM admin WHERE 1=1 UNION SELECT password"
    features = extract_query_features(sql, [])
    result = predict_anomaly(bundle, features)
    assert result["is_anomaly"] is True or result["anomaly_score"] > 0.3


def test_predict_output_keys(trained_model):
    bundle, _ = trained_model
    features = extract_query_features("Explain gradient descent", [])
    result = predict_anomaly(bundle, features)
    assert set(result.keys()) == {"is_anomaly", "anomaly_score", "classifier_prob", "isolation_score"}


def test_predict_score_in_range(trained_model):
    bundle, _ = trained_model
    features = extract_query_features("How does BERT work?", [])
    result = predict_anomaly(bundle, features)
    assert 0.0 <= result["classifier_prob"] <= 1.0


def test_metrics_n_features(trained_model):
    _, metrics = trained_model
    assert metrics["n_features"] == 15


def test_isolation_forest_in_bundle(trained_model):
    bundle, _ = trained_model
    from sklearn.ensemble import IsolationForest
    assert isinstance(bundle["isolation_forest"], IsolationForest)
