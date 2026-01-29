# Data Science Standards - Cheat Sheet

**Purpose**: Fast lookup for common decisions. Your quick reference card for algorithms, metrics, and solutions.

**Audience**: All data scientists (read this first before diving into detailed guides)

**Last Updated**: 2026-01-27

---

## 📋 Decision Tree: Which Guide Should I Read?

```
My current task is...
  |
  ├─ Starting a new ML project
  │   └─ → algorithm-selection-guidelines.md (Core Principles §1-10)
  |
  ├─ Choosing an algorithm
  │   ├─ Tabular data → classic-algorithms-reference.md (§1: Supervised Learning)
  │   ├─ Text/NLP → modern-algorithms-reference.md (§1: LLMs)
  │   ├─ Images → modern-algorithms-reference.md (§2: Vision Transformers)
  │   ├─ Time series → classic-algorithms-reference.md (§4: Time Series)
  │   ├─ Quantitative trading → quantitative-trading-guide.md (§2-6: Strategies)
  │   └─ Recommender → recommender-systems-guide.md (§2-4)
  |
  ├─ Designing features
  │   └─ → feature-engineering-patterns.md (§3-5: Patterns by data type)
  |
  ├─ Running A/B test or causal analysis
  │   └─ → experimentation-design-guide.md (§2-4: A/B testing, Causal inference)
  |
  ├─ Deploying to production
  │   ├─ Need monitoring → model-monitoring-guide.md (§3-5: Drift, Alerts)
  │   └─ Need optimization → model-deployment-guide.md (§2-3: Compression, Serving)
  |
  └─ Debugging poor performance
      ├─ Training issue → algorithm-selection-guidelines.md (§5.5: Debugging Tree)
      ├─ Production issue → model-monitoring-guide.md (§6: Incident Response)
      └─ Data issue → data-quality-validation-guide.md (§2-3)
```

---

## 🎯 Algorithm Selection (30-Second Guide)

### By Problem Type

| Problem | First Try | If Not Enough | SOTA (Complex) |
|---------|-----------|---------------|----------------|
| **Tabular Classification** | Logistic Regression | XGBoost | TabNet, Deep Nets |
| **Tabular Regression** | Linear/Ridge | XGBoost/LightGBM | Neural Networks |
| **Text Classification** | TF-IDF + LR | BERT fine-tuned | GPT-4, Claude |
| **Text Generation** | Template-based | GPT-3.5 | GPT-4, Claude 3.5 |
| **Image Classification** | ResNet pretrained | EfficientNet | ViT, CLIP |
| **Object Detection** | YOLO | Faster R-CNN | DETR, SAM |
| **Time Series Forecast** | ARIMA, Prophet | XGBoost | TimeGPT, TFT |
| **Quantitative Trading** | Moving Average | Mean Reversion, Pairs Trading | Reinforcement Learning, DL |
| **Recommender** | Matrix Factorization | Two-tower | Multi-task DL |
| **Clustering** | K-Means | DBSCAN | Hierarchical, GMM |
| **Anomaly Detection** | Isolation Forest | Autoencoder | Deep SVDD |

### By Data Size

| Sample Size | Recommended Approach |
|-------------|----------------------|
| < 1,000 | Linear models, shallow trees (max_depth ≤ 5) |
| 1K - 10K | Tree ensembles (RF, XGBoost), simple NNs |
| 10K - 100K | Gradient boosting, moderate NNs, fine-tuned transformers |
| 100K - 1M | Deep learning, large transformers |
| > 1M | Foundation models, self-supervised pretraining |

---

## 💡 Core Principles (One-Line Summary)

### Must-Know (Top 10)
1. **Occam's Razor**: Start simple (logistic → XGBoost → deep learning)
2. **No Free Lunch**: No algorithm dominates all problems—experiment!
3. **Data > Algorithms**: 10x more data often beats better algorithm
4. **Bias-Variance**: Balance model complexity vs generalization
5. **KISS**: Choose simpler model if accuracy difference < 2%
6. **GIGO**: Data quality is foundation—validate rigorously
7. **Cross-Validation**: Always use proper validation (never test on train)
8. **Feature Engineering**: Better features > fancier models
9. **Reproducibility**: Set seeds, log experiments, version data
10. **Baseline First**: Establish simple baseline before optimization

### Advanced (11-23)
11. **Bitter Lesson**: Scale (compute + data) beats hand-crafted features
12. **Ensemble Wisdom**: Average of models reduces variance
13. **Curse of Dimensionality**: High dims need exponentially more data
14. **Inductive Bias**: Match model assumptions to problem structure
15. **Explore vs Exploit**: Balance trying new things vs using what works
16. **Regularization as Prior**: L1/L2 = Bayesian priors on parameters
17. **Uncertainty Quantification**: Model should know when it's uncertain
18. **Correlation ≠ Causation**: Prediction needs correlation, decisions need causation
19. **Model Degradation**: All models decay—monitor and retrain
20. **Scaling Laws**: Performance follows power law with data/compute
21. **Transfer Learning**: Always start with pretrained models
22. **Representation Learning**: Good embeddings make everything easier
23. **Fairness**: Check performance across demographic groups

---

## 🔧 Common Problems & Solutions

### "My model overfits"
→ **Read**: algorithm-selection-guidelines.md §4 (Bias-Variance Tradeoff)
**Quick fix**: Increase regularization, reduce complexity, get more data

### "My A/B test is underpowered"
→ **Read**: experimentation-design-guide.md §3 (Power Analysis)
**Quick fix**: Calculate required sample size before starting test

### "Performance dropped in production"
→ **Read**: model-monitoring-guide.md §4 (Concept Drift Detection)
**Quick fix**: Check for data distribution shift, retrain with recent data

### "Model is too slow"
→ **Read**: model-deployment-guide.md §3 (Optimization Techniques)
**Quick fix**: Quantization (INT8), pruning, or use smaller model

### "Need to explain predictions"
→ **Read**: algorithm-selection-guidelines.md §25 (Explainability)
**Quick fix**: Use SHAP for tree models, LIME for black boxes

### "How do I build features?"
→ **Read**: feature-engineering-patterns.md §3-5
**Quick fix**: Time aggregations, ratios, interactions, embeddings

---

## 📊 Evaluation Metrics Quick Lookup

### Classification
| Metric | When to Use | Formula |
|--------|-------------|---------|
| **Accuracy** | Balanced classes | (TP+TN)/(P+N) |
| **Precision** | False positives costly | TP/(TP+FP) |
| **Recall** | False negatives costly | TP/(TP+FN) |
| **F1-Score** | Imbalanced classes | 2×(P×R)/(P+R) |
| **AUC-ROC** | Ranking quality | Area under ROC curve |
| **AUC-PR** | Severe imbalance | Area under Precision-Recall |

### Regression
| Metric | When to Use | Robust to Outliers? |
|--------|-------------|---------------------|
| **MAE** | Interpret error in original units | ✅ Yes |
| **RMSE** | Penalize large errors | ❌ No |
| **MAPE** | Percentage error | ❌ No |
| **R²** | Variance explained | ❌ No |

### Ranking (Recommender/Search)
| Metric | Meaning | Formula |
|--------|---------|---------|
| **Precision@K** | Relevant items in top K | relevant ∩ top-K / K |
| **Recall@K** | Coverage of relevant items | relevant ∩ top-K / \|relevant\| |
| **NDCG@K** | Ranking quality with position discount | DCG@K / Ideal-DCG@K |
| **MAP** | Mean Average Precision | Mean of Precision@i for relevant items |
| **MRR** | Mean Reciprocal Rank (first relevant) | 1 / rank of first relevant item |

---

## 🛒 Recommender Systems Quick Lookup

### By Scale & Data
| Scale | Data Type | Algorithm | Latency |
|-------|-----------|-----------|---------|
| **Small** (< 10K users) | Explicit ratings | SVD, User-based CF | Batch |
| **Medium** (10K - 1M) | Implicit feedback | ALS, Item-item CF | Batch/Real-time |
| **Large** (> 1M) | Implicit + features | Two-Tower + DCN | Real-time (< 100ms) |

### By Use Case
| Use Case | Retrieval | Ranking | Key Challenge |
|----------|-----------|---------|---------------|
| **E-commerce** | Item-Item CF | XGBoost | Cold start (new products) |
| **Video streaming** | Two-Tower | Multi-task (watch time + CTR) | Sequential behavior |
| **News** | Content-based | Contextual bandits | Freshness, diversity |
| **Social feeds** | Graph-based | GNN + engagement | Virality, network effects |

### Common Problems & Solutions
| Problem | Solution | Example |
|---------|----------|---------|
| **Cold start (new users)** | Popularity + demographics + onboarding quiz | TikTok diverse initial feed |
| **Cold start (new items)** | Content-based + early exploration | Netflix uses metadata |
| **Popularity bias** | Debiasing (inverse propensity weighting) | YouTube boosts long-tail |
| **Filter bubble** | Diversity constraints (MMR), serendipity | Spotify's Discovery Weekly |
| **Scalability** | Two-stage (retrieval + ranking) + FAISS | YouTube, Pinterest |

### Evaluation Checklist
```python
# Offline metrics
- NDCG@10 > 0.3 (good), > 0.4 (excellent)
- Coverage > 10% of catalog
- Intra-list diversity > 0.5

# Online metrics (A/B test)
- CTR improvement > 2%
- Watch time / dwell time increase
- Return rate (7-day retention)
```

---

## 📈 Quantitative Trading Quick Lookup

→ **Full Guide**: quantitative-trading-guide.md (comprehensive strategies and implementations)

### Strategy Selection (30-Second Guide)
| Strategy Type | Horizon | Algorithm | When to Use |
|---------------|---------|-----------|-------------|
| **Trend Following** | Days-Weeks | Moving Averages | Strong trending market |
| **Mean Reversion** | Hours-Days | Bollinger Bands, RSI | Range-bound market |
| **Statistical Arbitrage** | Minutes-Hours | Pairs Trading | High correlation pairs |
| **ML-Based** | Any | XGBoost, LSTM | Pattern-rich data |
| **Reinforcement Learning** | Any | PPO, DQN | Portfolio optimization |

### Key Performance Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| **Sharpe Ratio** | (R - Rf) / σ | > 1 (good), > 2 (excellent) |
| **Max Drawdown** | Max(Peak - Trough) / Peak | < 20% |
| **Win Rate** | Wins / Total Trades | > 50% |
| **Profit Factor** | Gross Profit / Gross Loss | > 1.5 |
| **Sortino Ratio** | (R - Rf) / Downside σ | > 1.5 |

### Essential Code Snippets

**Moving Average Crossover**:
```python
short_ma = df['close'].rolling(50).mean()
long_ma = df['close'].rolling(200).mean()
signal = (short_ma > long_ma).shift(1).astype(int)  # Avoid look-ahead bias
```

**Pairs Trading Z-Score**:
```python
from statsmodels.tsa.stattools import coint
score, pvalue, _ = coint(stock1, stock2)
spread = stock1 - hedge_ratio * stock2
zscore = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
```

**Vectorized Backtest**:
```python
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['signal'].shift(1) * df['returns']
df['cumulative'] = (1 + df['strategy_returns']).cumprod()
```

### Critical Pitfalls to Avoid
| Pitfall | Quick Fix |
|---------|----------|
| **Look-ahead bias** | Always `.shift(1)` signals |
| **Overfitting** | Walk-forward validation |
| **Survivorship bias** | Include delisted stocks |
| **No transaction costs** | Model 0.1-0.5% per trade |
| **Data snooping** | Reserve holdout period |

### Risk Management Checklist
- [ ] Position sizing (Kelly Criterion)
- [ ] Stop-loss: trailing or fixed
- [ ] Max drawdown limit < 20%
- [ ] Transaction cost: 0.1-0.5% per trade
- [ ] Walk-forward validation

→ **For detailed strategies, backtesting frameworks, RL implementations**: Read quantitative-trading-guide.md

---

## 🎓 When to Use Which Validation Strategy

| Data Type | Recommended Strategy | Why |
|-----------|----------------------|-----|
| **IID tabular** | K-Fold CV (k=5) | Assumes data is independent |
| **Imbalanced classes** | Stratified K-Fold | Maintains class distribution |
| **Time series** | Time Series Split | Respects temporal order |
| **Grouped data** (e.g., users) | Group K-Fold | Prevents leakage across groups |
| **Small dataset** (< 1K) | Leave-One-Out CV | Maximizes training data |
| **Large dataset** (> 100K) | Single holdout (80/20) | K-fold too expensive |

---

## 🚀 Production Checklist (Before Deployment)

### Must-Have
- [ ] Baseline comparison documented
- [ ] Cross-validation performed (no test leakage)
- [ ] Fairness checked across demographic groups
- [ ] Monitoring dashboards configured
- [ ] Rollback procedure defined
- [ ] Model card documentation complete
- [ ] Data drift detection enabled
- [ ] Alerting thresholds set

### Nice-to-Have
- [ ] Shadow deployment tested
- [ ] A/B test designed
- [ ] Explainability tools integrated
- [ ] Model compression applied
- [ ] Load testing completed

---

## 📁 File Navigation Guide

### Core Files (Read First)
| File | Size | When to Read | Key Sections |
|------|------|--------------|--------------|
| **algorithm-selection-guidelines.md** | 54KB | Every project start | §1-10 (Core Principles), §4 (Frameworks) |
| **classic-algorithms-reference.md** | 42KB | Choosing traditional ML | §1 (Supervised), Flowchart (§6) |
| **modern-algorithms-reference.md** | 46KB | Need SOTA/foundation models | §1 (LLMs), §2 (Vision) |

### Specialized Guides (Read When Needed)
| File | Size | When to Read |
|------|------|--------------|
| **experimentation-design-guide.md** | 35KB | Designing A/B test, causal inference |
| **feature-engineering-patterns.md** | 28KB | Building features, feature store |
| **model-monitoring-guide.md** | 32KB | Deploying to production, monitoring |
| **recommender-systems-guide.md** | 25KB | Building recommender systems |
| **model-deployment-guide.md** | 30KB | Optimization, compression, serving |
| **data-quality-validation-guide.md** | 22KB | Data validation, pipeline quality |

---

## 🔗 Cross-References (Related Topics)

| If You're Reading... | Also Check... |
|---------------------|---------------|
| Algorithm selection | → Feature engineering (better features often beat better models) |
| A/B testing | → Model monitoring (test setup similar to monitoring) |
| Feature engineering | → Data quality validation (features need clean data) |
| Model deployment | → Model monitoring (deployed models need monitoring) |
| Recommender systems | → Feature engineering (embeddings, interactions) |

---

## 💻 Code Snippets (Emergency Copy-Paste)

### Quick Baseline (Classification)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
print(f"Baseline F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Quick Baseline (Regression)
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Baseline MAE: {-scores.mean():.2f} ± {scores.std():.2f}")
```

### Detect Overfitting
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
gap = train_score - test_score

if gap > 0.1:
    print(f"⚠️ Overfitting detected! Train: {train_score:.3f}, Test: {test_score:.3f}")
    print("Solution: Increase regularization or get more data")
```

---

## 📞 When to Escalate (Agent Boundaries)

| Situation | Action |
|-----------|--------|
| **Unclear business requirements** | Escalate to stakeholders |
| **Insufficient data** | Escalate to data engineering team |
| **No feasible algorithm found** | Escalate to @data-scientist-tech-lead |
| **Ethical concerns** (bias, fairness) | Escalate to @data-scientist-tech-lead + ethics committee |
| **Infrastructure limitations** | Escalate to ML platform team |
| **Iteration limit exceeded** (>3 cycles) | Escalate to @data-scientist-tech-lead |

---

## 🎯 Quick Wins (High Impact, Low Effort)

1. **Use XGBoost for tabular data** (before trying neural networks)
2. **Fine-tune BERT instead of training from scratch** (NLP)
3. **Use pretrained ResNet/ViT** (computer vision)
6. **Backtest trading strategies with transaction costs** (quantitative trading)
7. **Create ratio features** (often more powerful than raw features)
8. **Use stratified CV for imbalanced classes**
9. **Set random seeds everywhere** (reproducibility)
10. **Start with simple model + great features** (not complex model + poor features)

---

## 🧭 Navigation Commands

```bash
# Search for specific topic
grep -r "topic" data-science-standards/

# List all algorithms by category
cat classic-algorithms-reference.md | grep "^###"

# Find all code examples
grep -A 10 "```python" *.md

# Get line numbers for specific section
grep -n "## Section Name" file.md
```

---

**Remember**: This is a quick reference. For deep understanding, read the full guides. When in doubt, start with `algorithm-selection-guidelines.md` Core Principles.

**Feedback**: If you find gaps or errors, update this file and the relevant detailed guides.
