# Data Science Standards Overview (Tech Lead Guide)

**Purpose**: Master index and navigation guide for data science standards, designed specifically for the Tech Lead role to coordinate and review all ML projects.

**Primary Audience**: Data Science Tech Lead  
**Secondary Audience**: Data scientists, ML engineers, research leads, algorithm designers, evaluators

**Last Updated**: 2026-01-27

---

## 📚 Documentation Structure

This directory contains all data science standards organized in a **3-tier system**:

### Tier 1: Quick Reference (Start Here)
- **cheat-sheet.md** (10 min read)
  - Fast decision trees for common scenarios
  - Algorithm selection in 30 seconds
  - Core principles one-liners
  - Emergency code snippets

### Tier 2: Core Standards (Read for Every Project)
- **algorithm-selection-guidelines.md** (60 min read)
  - 26 guiding principles (Occam's Razor to Fairness)
  - Mathematical foundations
  - Decision-making frameworks
  - Common pitfalls and anti-patterns

- **classic-algorithms-reference.md** (45 min read)
  - 31 proven algorithms (Linear Regression to A3C)
  - When to use, pros/cons, code examples
  - Supervised, unsupervised, deep learning, RL

- **modern-algorithms-reference.md** (50 min read)
  - 25 cutting-edge algorithms (GPT-4, CLIP, Diffusion, etc.)
  - LLMs, vision transformers, efficient training
  - 2024-2026 SOTA techniques

### Tier 3: Specialized Guides (Read When Needed)
- **experimentation-design-guide.md** (40 min read)
  - A/B testing methodology
  - Statistical power analysis
  - Causal inference techniques
  - Multiple testing corrections

- **feature-engineering-patterns.md** (35 min read)
  - Feature engineering by data type
  - Feature store architecture
  - Temporal features, embeddings, interactions
  - Real-world patterns from top tech companies

- **model-monitoring-guide.md** (40 min read)
  - Production monitoring strategies
  - Data drift and concept drift detection
  - Alerting and incident response
  - Observability best practices

- **recommender-systems-guide.md** (30 min read)
  - Collaborative filtering to neural recommendations
  - Two-tower models, multi-task learning
  - Cold start and session-based recommendations

- **quantitative-trading-guide.md** (45 min read)
  - Trading strategies (trend following, mean reversion, statistical arbitrage)
  - Machine learning for price prediction
  - Reinforcement learning for portfolio management
  - Backtesting frameworks and risk management

- **model-deployment-guide.md** (35 min read)
  - Model compression (quantization, pruning, distillation)
  - Serving patterns (batch, streaming, edge)
  - Optimization for inference

- **data-quality-validation-guide.md** (25 min read)
  - Schema and statistical validation
  - Data pipeline quality checks
  - Anomaly detection in data

---

## 🚀 Quick Start Guide

### For New Projects
1. Read: **cheat-sheet.md** (10 min)
2. Read: **algorithm-selection-guidelines.md** §1-10 Core Principles (20 min)
3. Use decision tree to find relevant algorithm in classic/modern references

### For A/B Testing
1. Read: **cheat-sheet.md** → A/B testing section
2. Read: **experimentation-design-guide.md** §2-4

### For Production Deployment
1. Read: **cheat-sheet.md** → Production checklist
2. Read: **model-monitoring-guide.md** §3-5
3. Read: **model-deployment-guide.md** §2-3 (if optimization needed)

### For Feature Engineering
1. Read: **cheat-sheet.md** → Feature problems section
2. Read: **feature-engineering-patterns.md** §3-5 (by your data type)

### For Quantitative Trading
1. Read: **cheat-sheet.md** → Quantitative Trading Quick Lookup
2. Read: **quantitative-trading-guide.md** §3-6 (classical to ML strategies)
3. Read: **quantitative-trading-guide.md** §8-9 (backtesting and risk management)

---

## 📖 How to Use These Standards

### For Data Science Research Leads
**Primary Reading**:
- algorithm-selection-guidelines.md (full)
- experimentation-design-guide.md §2-4
- cheat-sheet.md (always start here)

**Reference When Needed**:
- classic/modern-algorithms-reference.md (when researching algorithms)
- feature-engineering-patterns.md (when defining data requirements)

### For Data Science Algorithm Designers
**Primary Reading**:
- classic-algorithms-reference.md (full)
- modern-algorithms-reference.md (full)
- feature-engineering-patterns.md §3-5

**Reference When Needed**:
- algorithm-selection-guidelines.md (for guiding principles)
- experimentation-design-guide.md §5 (experiment design)

### For Data Scientists/Engineers
**Primary Reading**:
- cheat-sheet.md (always)
- algorithm-selection-guidelines.md §1-10
- feature-engineering-patterns.md (by data type)

**Reference When Needed**:
- model-deployment-guide.md (when deploying)
- model-monitoring-guide.md (when monitoring)

### For Data Science Evaluators
**Primary Reading**:
- experimentation-design-guide.md (full)
- model-monitoring-guide.md §4-6
- cheat-sheet.md → metrics section

**Reference When Needed**:
- algorithm-selection-guidelines.md §6 (evaluation philosophy)

---

## 🗂️ File Summaries

### cheat-sheet.md (10KB, ~500 lines)
**Purpose**: Fast lookup for 90% of common decisions

**Key Sections**:
- Decision trees: "Which guide should I read?"
- Algorithm selection by problem type and data size
- Core principles in one line each
- Common problems & quick solutions
- Evaluation metrics quick lookup
- Production deployment checklist
- Emergency code snippets

**When to Use**: Start of every task, when stuck, before reading detailed guides

---

### algorithm-selection-guidelines.md (54KB, ~1760 lines)
**Purpose**: Philosophical foundations and guiding principles for ML

**Key Sections**:
1. Core Principles (1-10): Occam's Razor, No Free Lunch, Data > Algorithms, etc.
2. Advanced Principles (11-23): Bitter Lesson, Ensemble Wisdom, Scaling Laws, etc.
3. Mathematical Foundations: Bias-Variance, Information Theory, PAC Learning
4. Decision-Making Frameworks: Model selection trees, complexity tradeoffs
5. Common Pitfalls: Data leakage, selection bias, p-hacking, etc.
6. Ethical Considerations: Fairness, transparency, privacy
7. Philosophy in Practice: Scientific method for ML, debugging manifesto

**When to Use**: 
- Starting any new project (read §1-10)
- Making algorithm selection decisions
- Debugging model performance issues
- Ethical/fairness reviews

**Key Quotes**:
- "All models are wrong, but some are useful" — George Box
- "More data beats clever algorithms" — Unreasonable Effectiveness of Data
- "It's not who has the best algorithm that wins. It's who has the most data." — Andrew Ng

---

### classic-algorithms-reference.md (42KB, ~1570 lines)
**Purpose**: Reference guide for proven, time-tested algorithms

**Coverage**: 31 algorithms across:
- Supervised Learning: Regression (Linear, Ridge, Lasso, Elastic Net, SVR)
- Classification: Logistic, SVM, Naive Bayes, KNN, Decision Trees
- Ensemble Methods: Random Forest, Gradient Boosting, XGBoost, AdaBoost
- Unsupervised: K-Means, DBSCAN, Hierarchical Clustering, GMM
- Dimensionality Reduction: PCA, t-SNE, UMAP
- Time Series: ARIMA, Prophet, SARIMAX
- Deep Learning: CNN, RNN, LSTM, GRU, Attention
- Reinforcement Learning: Q-Learning, SARSA, DQN, Policy Gradient, Actor-Critic, DDPG, A2C/A3C

**Format**: Each algorithm includes:
- When to use
- Pros and cons
- Python code example
- When NOT to use
- Real-world applications

**Key Feature**: Algorithm selection flowchart (§6, line 1534)

**When to Use**:
- Choosing algorithm for tabular data
- Need interpretable model
- Working with small/medium datasets (< 100K samples)
- Baseline establishment

---

### modern-algorithms-reference.md (46KB, ~1625 lines)
**Purpose**: Cutting-edge algorithms and architectures (2024-2026)

**Coverage**: 25 modern algorithms across:
- Large Language Models: GPT-4, Claude 3, Llama 3, Mistral, Gemini
- Vision & Multimodal: ViT, CLIP, SAM, DINOv2
- Diffusion Models: Stable Diffusion 3, DALL-E 3, Consistency Models
- Efficient Training: LoRA, QLoRA, Flash Attention, GQA
- Graph Neural Networks: Graph Transformers
- Modern RL: PPO, DPO, SAC, TD3, CQL, Decision Transformer, MARL, RLHF alternatives
- Time Series Foundation Models: TimeGPT, Lag-Llama, TFT
- AutoML: AutoGluon, H2O AutoML
- Emerging: RAG, Mamba (State Space Models)

**Key Trends (2024-2026)**:
- Mixture of Experts (MoE) scaling
- Multimodal models becoming standard
- Efficient fine-tuning (LoRA, QLoRA)
- Open-source LLMs competitive with proprietary
- Alignment beyond RLHF (DPO, RLAIF)

**When to Use**:
- Need state-of-the-art performance
- NLP, computer vision, multimodal tasks
- Large datasets (> 100K samples)
- Access to GPU resources

---

### experimentation-design-guide.md (35KB, ~1200 lines)
**Purpose**: Rigorous experimental methodology for data science

**Key Sections**:
1. A/B Testing Fundamentals
   - Randomization strategies
   - Statistical power analysis
   - Sample size calculation
   - Confidence intervals

2. Advanced Experimental Designs
   - Multi-armed bandits (Thompson Sampling, UCB)
   - Sequential testing
   - Switchback experiments
   - Stratified experiments

3. Causal Inference
   - Propensity Score Matching
   - Difference-in-Differences
   - Regression Discontinuity
   - Instrumental Variables
   - Synthetic Control
   - DoWhy framework

4. Multiple Testing Corrections
   - Bonferroni correction
   - Benjamini-Hochberg (FDR)
   - When and how to use

5. Real-World Considerations
   - Network effects
   - Interference between units
   - Novelty effects
   - Seasonality handling

**When to Use**:
- Designing A/B tests
- Measuring causal impact
- Validating model improvements
- Making go/no-go decisions

**Key Tools**: scipy.stats, statsmodels, dowhy, causalml

---

### feature-engineering-patterns.md (28KB, ~950 lines)
**Purpose**: Practical patterns for feature engineering at scale

**Key Sections**:
1. Feature Store Architecture
   - Online vs offline features
   - Point-in-time correctness
   - Feature versioning
   - Freshness vs staleness tradeoffs

2. Feature Engineering by Data Type
   - **Tabular**: Binning, one-hot encoding, target encoding, ratios
   - **Text**: TF-IDF, word embeddings, sentence embeddings
   - **Time Series**: Lags, rolling statistics, seasonal decomposition
   - **Images**: Pretrained embeddings, augmentations
   - **Graphs**: Node embeddings, graph statistics

3. Advanced Patterns
   - Interaction features
   - Polynomial features
   - Embeddings from categorical variables
   - Feature crosses

4. Feature Selection
   - Correlation-based
   - Model-based (feature importance)
   - Wrapper methods
   - L1 regularization

5. Production Patterns
   - Feature serving (real-time vs batch)
   - Feature monitoring
   - Handling feature drift

**Real-World Examples**:
- Uber Michelangelo feature store
- Airbnb Zipline
- LinkedIn Frame

**When to Use**:
- Designing feature engineering pipelines
- Building feature stores
- Debugging feature quality issues

---

### model-monitoring-guide.md (32KB, ~1100 lines)
**Purpose**: Production ML monitoring and observability

**Key Sections**:
1. Monitoring Strategy
   - What to monitor (model, data, system)
   - Monitoring layers (input, output, performance)
   - SLIs, SLOs, SLAs for ML systems

2. Data Drift Detection
   - Statistical tests (KS test, Chi-square)
   - Population Stability Index (PSI)
   - KL divergence
   - When to retrain

3. Model Performance Monitoring
   - Online vs offline metrics
   - Prediction distribution shift
   - Calibration monitoring
   - Segment-level performance

4. Concept Drift Detection
   - Supervised drift detection
   - Unsupervised drift detection
   - Drift severity scoring

5. Alerting and Incident Response
   - Alert thresholds
   - Alert fatigue prevention
   - Incident response runbooks
   - Automated rollback procedures

6. Observability Tools
   - Prometheus, Grafana
   - MLflow, Weights & Biases
   - Custom dashboards

**When to Use**:
- Deploying models to production
- Investigating production issues
- Designing retraining strategies

**Key Insight**: "All models degrade over time—monitoring is not optional"

---

### recommender-systems-guide.md (25KB, ~850 lines)
**Purpose**: Specialized guide for recommendation systems

**Key Sections**:
1. Recommender System Types
   - Content-based filtering
   - Collaborative filtering (user-based, item-based)
   - Hybrid approaches

2. Classical Algorithms
   - Matrix Factorization (ALS, SVD)
   - Factorization Machines
   - BPR (Bayesian Personalized Ranking)

3. Neural Recommenders
   - Two-tower models (Dual Encoder)
   - Neural Collaborative Filtering (NCF)
   - Deep & Cross Network (DCN)
   - Wide & Deep

4. Advanced Techniques
   - Multi-task learning (CTR + Conversion)
   - Session-based (GRU4Rec)
   - Sequential recommendations
   - Context-aware recommendations

5. Production Challenges
   - Cold start problem (new users/items)
   - Popularity bias
   - Filter bubble avoidance
   - Real-time serving at scale

**Real-World Examples**:
- YouTube recommendations
- Netflix recommendations
- TikTok For You page
- Amazon product recommendations

**When to Use**:
- Building recommendation systems
- Improving recommendation quality
- Handling cold start problems

---

### model-deployment-guide.md (30KB, ~1000 lines)
**Purpose**: Optimization and deployment best practices

**Key Sections**:
1. Model Compression Techniques
   - **Quantization**: INT8, INT4, GPTQ, AWQ
   - **Pruning**: Structured vs unstructured
   - **Knowledge Distillation**: Teacher-student training
   - **Low-rank approximation**: LoRA, SVD

2. Serving Patterns
   - Batch inference
   - Online (real-time) inference
   - Streaming inference
   - Edge deployment

3. Optimization for Inference
   - ONNX conversion
   - TensorRT optimization
   - torch.compile
   - Flash Attention

4. Deployment Strategies
   - Blue-green deployment
   - Canary deployment
   - Shadow deployment
   - A/B testing in production

5. Cost Optimization
   - Spot instances
   - Auto-scaling
   - Model caching
   - Batch size tuning

**Tools**: ONNX, TensorRT, TorchServe, BentoML, Ray Serve

**When to Use**:
- Model is too large or slow
- Need to reduce inference costs
- Deploying to edge devices
- Scaling to high QPS

---

### data-quality-validation-guide.md (22KB, ~750 lines)
**Purpose**: Ensuring data quality throughout ML pipelines

**Key Sections**:
1. Data Validation Layers
   - Schema validation
   - Statistical validation
   - Semantic validation

2. Validation Techniques
   - Range checks
   - Distribution checks
   - Consistency checks
   - Completeness checks

3. Tools and Frameworks
   - Great Expectations
   - TensorFlow Data Validation (TFDV)
   - Deequ (Amazon)
   - Pandera

4. Data Lineage and Governance
   - Tracking data provenance
   - Data versioning
   - Compliance (GDPR, CCPA)

5. Anomaly Detection in Data
   - Statistical methods
   - ML-based detection
   - Real-time vs batch

**When to Use**:
- Building data pipelines
- Debugging data quality issues
- Ensuring compliance

---

## 🔄 How Standards Evolve

These standards are living documents that evolve with:
- New algorithm developments
- Production learnings
- Industry best practices
- Team feedback

**Update Frequency**:
- Core standards: Quarterly review
- Algorithm references: Updated with major releases
- Quick reference: As needed

**Contribution Process**:
1. Identify gap or outdated content
2. Create issue or proposal
3. Update relevant files
4. Update cheat-sheet.md if navigation changes
5. Update this README.md if new file added

---

## 🏗️ Design Principles

These standards follow these design principles:

1. **Layered Information**: Quick → Core → Specialized
2. **Action-Oriented**: Focus on "when to use" not just "what is"
3. **Example-Rich**: Every concept has code examples
4. **Cross-Referenced**: Related topics link to each other
5. **Industry-Proven**: Based on practices from Google, Meta, OpenAI, etc.
6. **Tool-Agnostic** (where possible): Principles over specific tools
7. **Reproducible**: All examples can be run

---

## 📞 Support and Feedback

**Questions about standards?**
- Check cheat-sheet.md first
- Search within relevant guide
- Ask @data-scientist-tech-lead

**Found an error or gap?**
- Create issue with file name and line number
- Suggest improvement
- Update and submit for review

**Want to contribute?**
- Follow the contribution process above
- Ensure examples are tested
- Maintain consistent format

---

## 📚 External Resources

These standards are informed by:
- Industry: Google, Meta, OpenAI, Microsoft, Netflix, Uber, Airbnb
- Academia: Top conferences (NeurIPS, ICML, KDD, ACL, CVPR)
- Open source: Hugging Face, Papers with Code, GitHub
- Books: "Designing Machine Learning Systems" (Chip Huyen), "ML System Design Interview" (Ali Aminian)

---

**Version**: 1.0  
**Last Updated**: 2026-01-27  
**Maintainer**: Data Science Team  
**License**: Internal Use Only
