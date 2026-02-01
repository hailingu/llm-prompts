# Model Monitoring Guide

**Purpose**: Production ML monitoring and observability — ensuring models remain accurate, reliable, and fair after deployment.

**Target Audience**: ML engineers, data scientists, SREs

**Last Updated**: 2026-01-27

---

## Why Monitoring Matters

### The Inevitable Decay

**Fact**: All ML models degrade over time in production.

**Reasons**:
1. **Data Distribution Shift**: Input data changes (new user demographics, device types, etc.)
2. **Concept Drift**: Relationship between features and target changes (user behavior evolves)
3. **Upstream Data Issues**: Bugs in data pipeline, missing features, schema changes
4. **Adversarial Behavior**: Fraud patterns evolve to evade detection
5. **Seasonality**: Holiday shopping patterns, weather changes, etc.

**Without monitoring**: Models silently fail, causing business damage

---

## What to Monitor: 3-Layer Framework

```text
Layer 1: System Metrics (Infrastructure)
   ├─ Latency (p50, p95, p99)
   ├─ Throughput (QPS)
   ├─ Error rates (4xx, 5xx)
   └─ Resource usage (CPU, memory, GPU)

Layer 2: Data Metrics (Input Quality)
   ├─ Feature distributions (mean, std, min, max)
   ├─ Missing value rates
   └─ Data drift scores (PSI, KL divergence)

Layer 3: Model Metrics (Output Quality)
   ├─ Prediction distribution (mean, std, entropy)
   ├─ Confidence scores
   ├─ Performance metrics (accuracy, F1, AUC)
   └─ Business metrics (revenue, conversions, churn)
```

---

## Data Drift Detection

### Method 1: Population Stability Index (PSI)

**Most popular method** for monitoring feature drift

**Formula**:
```
PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
```

**Interpretation**:
- PSI < 0.1: No significant change
- 0.1 ≤ PSI < 0.2: Moderate drift (investigate)
- PSI ≥ 0.2: Significant drift (retrain!)

```python
import numpy as np

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index (PSI)"""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    
    return psi
```

---

### Method 2: KL Divergence

```python
from scipy.stats import entropy

def calculate_kl_divergence(expected, actual, bins=50):
    """Calculate KL divergence between expected and actual distributions"""
    range_min = min(expected.min(), actual.min())
    range_max = max(expected.max(), actual.max())
    
    expected_hist, _ = np.histogram(expected, bins=bins, range=(range_min, range_max))
    actual_hist, _ = np.histogram(actual, bins=bins, range=(range_min, range_max))
    
    # Normalize
    expected_hist = expected_hist / expected_hist.sum()
    actual_hist = actual_hist / actual_hist.sum()
    
    # Avoid zeros
    expected_hist = np.where(expected_hist == 0, 1e-10, expected_hist)
    actual_hist = np.where(actual_hist == 0, 1e-10, actual_hist)
    
    kl = entropy(actual_hist, expected_hist)
    return kl
```

---

### Method 3: Kolmogorov-Smirnov Test

```python
from scipy.stats import ks_2samp

def detect_drift_ks(expected, actual, alpha=0.05):
    """Kolmogorov-Smirnov test for distribution drift"""
    statistic, p_value = ks_2samp(expected, actual)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'drift_detected': p_value < alpha
    }
```

---

## Comprehensive Drift Monitor

```python
class DriftMonitor:
    def __init__(self, reference_data):
        """Initialize drift monitor with reference (training) data"""
        self.reference = reference_data
        self.feature_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, data):
        stats = {}
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                stats[col] = {
                    'type': 'continuous',
                    'values': data[col].values
                }
            else:
                stats[col] = {
                    'type': 'categorical',
                    'counts': data[col].value_counts()
                }
        return stats
    
    def detect_drift(self, current_data):
        """Detect drift in current data compared to reference"""
        drift_report = {}
        
        for col in current_data.columns:
            if col not in self.feature_stats:
                continue
            
            if self.feature_stats[col]['type'] == 'continuous':
                psi = calculate_psi(
                    self.feature_stats[col]['values'],
                    current_data[col].values
                )
                drift_report[col] = {
                    'method': 'PSI',
                    'score': psi,
                    'drift': psi >= 0.2
                }
        
        return drift_report

# Usage
monitor = DriftMonitor(training_data)
drift_report = monitor.detect_drift(production_data)

for feature, result in drift_report.items():
    if result['drift']:
        print(f"⚠️ DRIFT detected in {feature}: {result}")
```

---

## Model Performance Monitoring

### Prediction Distribution Monitoring

```python
def monitor_predictions(train_preds, prod_preds):
    """Monitor prediction distribution shift"""
    from scipy.stats import ks_2samp
    
    ks_stat, p_value = ks_2samp(train_preds, prod_preds)
    
    print(f"Training mean: {train_preds.mean():.3f}")
    print(f"Production mean: {prod_preds.mean():.3f}")
    print(f"KS test p-value: {p_value:.4f}")
    
    if p_value < 0.01:
        print("⚠️ Prediction distribution has shifted!")
```

---

### Confidence-Based Monitoring

```python
def monitor_confidence(predictions, threshold=0.6):
    """Monitor prediction confidence"""
    # For binary classification
    confidence = np.maximum(predictions, 1 - predictions)
    
    low_confidence_ratio = (confidence < threshold).mean()
    
    print(f"Mean confidence: {confidence.mean():.3f}")
    print(f"Low confidence ratio (< {threshold}): {low_confidence_ratio:.2%}")
    
    if low_confidence_ratio > 0.2:
        print("⚠️ High proportion of low-confidence predictions!")
    
    return low_confidence_ratio
```

---

### Calibration Monitoring

```python
from sklearn.calibration import calibration_curve

def monitor_calibration(y_true, y_pred, n_bins=10):
    """Monitor model calibration"""
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    
    if calibration_error > 0.1:
        print("⚠️ Model is poorly calibrated!")
    
    return calibration_error
```

---

## Concept Drift Detection

### Performance-Based Detection

```python
def detect_concept_drift_performance(dates, y_true, y_pred, window_size=7):
    """Detect concept drift via performance degradation"""
    from sklearn.metrics import f1_score
    
    daily_f1 = []
    
    for date in pd.date_range(dates.min(), dates.max()):
        mask = (dates >= date - pd.Timedelta(days=window_size)) & (dates <= date)
        
        if mask.sum() > 100:
            f1 = f1_score(y_true[mask], (y_pred[mask] > 0.5).astype(int))
            daily_f1.append({'date': date, 'f1': f1})
    
    df_perf = pd.DataFrame(daily_f1)
    
    baseline_f1 = df_perf['f1'].iloc[:30].mean()
    current_f1 = df_perf['f1'].iloc[-7:].mean()
    
    degradation = (baseline_f1 - current_f1) / baseline_f1
    
    if degradation > 0.05:
        print(f"⚠️ Concept drift! Performance degraded {degradation:.1%}")
    
    return df_perf
```

---

### Unsupervised Drift Detection

```python
def detect_drift_unsupervised(X_ref, X_curr):
    """Detect drift without labels using classifier approach"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    # Label reference as 0, current as 1
    y_ref = np.zeros(len(X_ref))
    y_curr = np.ones(len(X_curr))
    
    X_combined = np.vstack([X_ref, X_curr])
    y_combined = np.hstack([y_ref, y_curr])
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    auc_scores = cross_val_score(clf, X_combined, y_combined, cv=5, scoring='roc_auc')
    auc_mean = auc_scores.mean()
    
    # AUC ~ 0.5: No drift, AUC > 0.7: Moderate drift, AUC > 0.9: Significant drift
    
    if auc_mean > 0.7:
        print("⚠️ Drift detected!")
        clf.fit(X_combined, y_combined)
        print("Top drifting features:", 
              pd.Series(clf.feature_importances_).sort_values(ascending=False).head())
    
    return auc_mean
```

---

## Alert Thresholds

```python
ALERT_THRESHOLDS = {
    # Data drift
    'psi': {'warning': 0.1, 'critical': 0.2},
    
    # Performance degradation
    'f1_degradation': {'warning': 0.05, 'critical': 0.10},
    
    # Confidence
    'low_confidence_ratio': {'warning': 0.15, 'critical': 0.25},
    
    # System
    'latency_p95_ms': {'warning': 200, 'critical': 500},
    'error_rate': {'warning': 0.01, 'critical': 0.05}
}
```

---

## Smart Alerting (Adaptive Thresholds)

```python
def should_alert(metric_name, current_value, historical_values, sensitivity='medium'):
    """Smart alerting with adaptive thresholds"""
    mean = np.mean(historical_values)
    std = np.std(historical_values)
    
    thresholds = {'low': 3.0, 'medium': 2.5, 'high': 2.0}
    threshold = thresholds[sensitivity]
    z_score = abs((current_value - mean) / std)
    
    if z_score > threshold:
        return {
            'alert': True,
            'severity': 'critical' if z_score > threshold + 1 else 'warning',
            'message': f"{metric_name} is {z_score:.1f} std devs from mean"
        }
    
    return {'alert': False}
```

---

## Monitoring Pipeline Example

```python
def monitoring_pipeline():
    """Daily monitoring pipeline"""
    # 1. Fetch data
    prod_data = fetch_production_data(last_n_days=1)
    train_data = fetch_training_data()
    
    # 2. Compute drift metrics
    drift_metrics = {}
    for feature in prod_data.columns:
        psi = calculate_psi(train_data[feature], prod_data[feature])
        drift_metrics[feature] = psi
    
    # 3. Check drift thresholds
    for feature, psi in drift_metrics.items():
        if psi > 0.2:
            send_alert('Data drift', f'Feature {feature} has PSI {psi:.3f}')
    
    # 4. Fetch ground truth (if available)
    labels = fetch_labels(prod_data['request_id'])
    if not labels.empty:
        f1 = compute_f1(labels, prod_data['prediction'])
        
        baseline_f1 = get_baseline_metric('f1_score')
        if f1 < baseline_f1 * 0.95:
            send_alert('Performance degradation', 
                      f'F1 dropped from {baseline_f1:.3f} to {f1:.3f}')
    
    # 5. Update dashboards
    update_dashboard({'date': datetime.now(), 'drift_metrics': drift_metrics})
```

---

## Tools and Platforms

### Open Source

| Tool                     | Purpose                       |
| ------------------------ | ----------------------------- |
| ------                   | ---------                     |
| **Evidently AI**         | Drift detection, data quality |
| **Great Expectations**   | Data validation               |
| **Prometheus + Grafana** | Metrics + visualization       |
| **MLflow**               | Experiment tracking           |
| **WhyLabs**              | ML observability              |

### Example: Evidently AI

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)

# Save HTML report
report.save_html("drift_report.html")

# Get metrics programmatically
metrics = report.as_dict()
drift_share = metrics['metrics'][0]['result']['drift_share']

if drift_share > 0.3:
    print(f"⚠️ {drift_share:.1%} of features have drift!")
```

---

## Monitoring Checklist

### Before Deployment
- [ ] Define metrics to monitor (data, model, business)
- [ ] Set up logging (features, predictions, metadata)
- [ ] Define alert thresholds
- [ ] Create monitoring dashboards
- [ ] Write incident response runbook
- [ ] Test alerting system

### After Deployment
- [ ] Monitor daily for first week
- [ ] Weekly drift reports
- [ ] Monthly performance review
- [ ] Quarterly model refresh evaluation

### When Alert Fires
- [ ] Check runbook
- [ ] Investigate data drift first
- [ ] Check recent changes
- [ ] Segment analysis
- [ ] Decide: rollback, retrain, or tune
- [ ] Document learnings

---

**Key Takeaway**: **All models degrade—monitoring is not optional**. Set up monitoring before deployment, not after incidents.

**Version**: 1.0  
**Maintainer**: Data Science Team
