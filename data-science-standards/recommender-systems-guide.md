# Recommender Systems Guide

**Purpose**: Comprehensive guide for building production recommendation systems, from classical collaborative filtering to modern neural architectures.

**Audience**: Data scientists, ML engineers designing and implementing recommendation systems

**Last Updated**: 2026-01-27

---

## Table of Contents
1. [Fundamentals](#1-fundamentals)
2. [Classical Approaches](#2-classical-approaches)
3. [Neural Recommenders](#3-neural-recommenders)
4. [Advanced Techniques](#4-advanced-techniques)
5. [Evaluation & Metrics](#5-evaluation--metrics)
6. [Production Challenges](#6-production-challenges)
7. [Architecture Patterns](#7-architecture-patterns)
8. [Case Studies](#8-case-studies)

---

## 1. Fundamentals

### 1.1 Problem Definition

**Recommendation as a Prediction Task**:
- **Input**: User, item, context
- **Output**: 
  - **Rating prediction**: Predict rating user will give to item (e.g., 1-5 stars)
  - **Ranking**: Order items by relevance for user
  - **Top-K retrieval**: Select K most relevant items from millions

**Three-Stage Pipeline** (Modern Production Systems):
1. **Candidate Generation (Retrieval)**: Reduce millions of items → hundreds (fast, recall-focused)
2. **Ranking**: Rank hundreds of candidates by relevance (slow, precision-focused)
3. **Re-ranking**: Apply business rules, diversity, freshness

### 1.2 Types of Feedback

| Feedback Type | Examples | Characteristics | Algorithm Choice |
|---------------|----------|-----------------|------------------|
| **Explicit** | Ratings (1-5 stars), thumbs up/down | Sparse, accurate | SVD, PMF, NCF |
| **Implicit** | Clicks, views, watch time, purchases | Dense, noisy | ALS, BPR, Two-Tower |
| **Contextual** | Time, location, device, session | Rich features | FM, DCN, Transformers |

**Implicit > Explicit in Production**: Most systems use implicit feedback (abundant data)

### 1.3 Key Challenges

1. **Cold Start Problem**
   - **New users**: No historical behavior → Use demographics, onboarding quiz, popularity
   - **New items**: No interactions → Use content features, early exploration (bandits)

2. **Scalability**
   - Billions of user-item pairs (10⁶ users × 10⁷ items = 10¹³ pairs)
   - Real-time serving (< 100ms latency)

3. **Data Sparsity**
   - Most users interact with < 1% of items
   - User-item matrix is 99.9% zeros

4. **Popularity Bias**
   - Algorithms over-recommend popular items
   - Long-tail items get ignored → Diversity metrics needed

5. **Filter Bubble**
   - Users only see similar items → Echo chamber effect
   - Solution: Exploration-exploitation, serendipity

---

## 2. Classical Approaches

### 2.1 Content-Based Filtering

**Core Idea**: Recommend items similar to what user liked in the past

**Algorithm**:
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def fit(self, item_features):
        """
        item_features: List of item descriptions (text)
        """
        # Convert text to TF-IDF vectors
        self.item_vectors = self.vectorizer.fit_transform(item_features)
        self.item_similarity = cosine_similarity(self.item_vectors)
    
    def recommend(self, user_liked_items, n=10):
        """
        user_liked_items: List of item indices user liked
        """
        # Average similarity to all items user liked
        scores = self.item_similarity[user_liked_items].mean(axis=0)
        
        # Remove already liked items
        scores[user_liked_items] = -1
        
        # Top-N
        top_indices = np.argsort(scores)[::-1][:n]
        return top_indices, scores[top_indices]

# Example: Movie recommendations
movie_descriptions = [
    "Action thriller with car chases",
    "Romantic comedy about wedding",
    "Sci-fi space adventure",
    "Action movie with explosions",
    "Romantic drama about loss",
]

cb_rec = ContentBasedRecommender()
cb_rec.fit(movie_descriptions)

# User liked movies 0 and 3 (both action)
recommendations, scores = cb_rec.recommend(user_liked_items=[0, 3], n=3)
print(f"Recommended movies: {recommendations}")
print(f"Similarity scores: {scores}")
```

**Pros**:
- ✅ No cold start for items (use content features immediately)
- ✅ Explainable: "You liked X because it's similar to Y"
- ✅ No popularity bias

**Cons**:
- ❌ Requires item features (metadata, text, images)
- ❌ Over-specialization: Only recommends similar items (no diversity)
- ❌ Cannot capture collaborative patterns (what others like)

**When to Use**:
- Rich item metadata available (news articles, products with descriptions)
- Cold start for new items is critical
- Need explainability

---

### 2.2 Collaborative Filtering

**Core Idea**: Users who agreed in the past will agree in the future

See [classic-algorithms-reference.md](classic-algorithms-reference.md#30-collaborative-filtering-user-based--item-based) for full implementation.

**Quick Comparison**:

| Approach | How It Works | Scalability | Cold Start |
|----------|-------------|-------------|------------|
| **User-based CF** | Find similar users, recommend what they liked | O(n_users²) | Poor for new users |
| **Item-based CF** | Find similar items, recommend similar to what user liked | O(n_items²) | Poor for new items |
| **Matrix Factorization** | Learn latent user/item embeddings | O(k×interactions) | Poor for both |

**Production Recommendation**: Item-based CF (more stable than user-based, easier to explain)

---

### 2.3 Matrix Factorization

**Core Idea**: Decompose user-item matrix into low-rank factors

See [classic-algorithms-reference.md](classic-algorithms-reference.md#31-matrix-factorization-svd-als-nmf) for full implementation.

**Key Variants**:
- **SVD**: Classic, for explicit ratings
- **ALS (Alternating Least Squares)**: For implicit feedback, scalable
- **BPR (Bayesian Personalized Ranking)**: Optimizes ranking directly
- **NMF (Non-negative Matrix Factorization)**: Interpretable factors

**Netflix Prize Winner (2009)**: Ensemble of SVD variants

```python
# Quick ALS example
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

# user_item_matrix: sparse matrix (n_users, n_items)
model = AlternatingLeastSquares(factors=100, iterations=15)
model.fit(user_item_matrix)

# Recommend for user
recommendations = model.recommend(
    userid=user_id,
    user_items=user_item_matrix[user_id],
    N=10
)
```

**When to Use**:
- Large-scale implicit feedback (millions of users/items)
- Baseline for recommendation systems
- Need user/item embeddings for downstream tasks

---

## 3. Neural Recommenders

### 3.1 Two-Tower Model (Dual Encoder)

**Architecture**: Separate neural networks for user and item, dot product for similarity

See [modern-algorithms-reference.md](modern-algorithms-reference.md#24-two-tower-model-dual-encoder-for-recommendations) for full implementation.

**Use Case**: **Candidate generation** (retrieve top 100-1000 from millions)

**Production Serving**:
1. **Offline**: Compute item embeddings for all items → index in FAISS
2. **Online**: Compute user embedding → ANN search in item index (< 10ms)

```python
# Inference with FAISS
import faiss

# Build index (offline)
item_embeddings = model.get_all_item_embeddings()  # (n_items, 128)
index = faiss.IndexFlatIP(128)  # Inner product search
index.add(item_embeddings)

# Online serving
user_embedding = model.get_user_embedding(user_id)  # (1, 128)
scores, item_ids = index.search(user_embedding, k=100)
```

**Real-World**: YouTube, TikTok, Pinterest

---

### 3.2 Neural Collaborative Filtering (NCF)

**Architecture**: Combines matrix factorization (GMF) and multi-layer perceptron (MLP)

See [modern-algorithms-reference.md](modern-algorithms-reference.md#26-neural-collaborative-filtering-ncf) for full implementation.

**Key Insight**: Learn nonlinear user-item interactions (better than dot product)

**Comparison**:
- **Matrix Factorization**: `score = user_vec · item_vec` (linear)
- **NCF**: `score = MLP(user_vec, item_vec)` (nonlinear)

**When to Use**:
- Have abundant interaction data (> 100K interactions)
- Matrix factorization plateaus in performance
- Ranking stage (not retrieval)

---

### 3.3 Deep & Cross Network (DCN)

**Architecture**: Cross network (explicit polynomial feature crosses) + deep network

See [modern-algorithms-reference.md](modern-algorithms-reference.md#25-deep--cross-network-dcn) for full implementation.

**Use Case**: **CTR prediction** with sparse categorical features

**Key Innovation**: Cross network learns bounded-degree polynomial feature interactions efficiently

```
Cross layer: x_{l+1} = x0 ⊗ (w_l^T x_l) + b_l + x_l
```

**When to Use**:
- Ranking stage (after candidate generation)
- Have user/item features (demographics, categories)
- Need explicit feature crosses (user_age × item_category)

**Real-World**: Google Ads, Facebook Ads ranking

---

### 3.4 Wide & Deep

**Architecture**: Wide (memorization) + Deep (generalization)

**Difference from DCN**: Wide = manual feature crosses, DCN = automatic crosses

```python
# PyTorch implementation
class WideAndDeep(nn.Module):
    def __init__(self, wide_dim, deep_input_dim, deep_layers=[128, 64]):
        super().__init__()
        
        # Wide part: linear model on manual crosses
        self.wide = nn.Linear(wide_dim, 1)
        
        # Deep part: DNN
        layers = []
        in_dim = deep_input_dim
        for hidden_dim in deep_layers:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*layers)
    
    def forward(self, wide_features, deep_features):
        wide_out = self.wide(wide_features)
        deep_out = self.deep(deep_features)
        return torch.sigmoid(wide_out + deep_out)
```

**When to Use**:
- Have domain knowledge for manual feature crosses
- Need balance between memorization (popularity) and generalization (personalization)

**Real-World**: Google Play Store recommendations (original paper, 2016)

---

## 4. Advanced Techniques

### 4.1 Sequential Recommendations

**Goal**: Model user behavior as a sequence (session-based, next-item prediction)

**Key Algorithms**:

1. **GRU4Rec** (2016): Use GRU to model session sequences
2. **SASRec** (2018): Self-attention for sequential recommendations
3. **BERT4Rec** (2019): Bidirectional encoding of user history

**SASRec Implementation**:

```python
import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)

class SASRec(nn.Module):
    def __init__(self, n_items, hidden_dim=128, num_layers=2, max_len=50):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        
        self.attention_layers = nn.ModuleList([
            SelfAttentionLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, n_items)
    
    def forward(self, item_seq):
        """
        item_seq: (batch_size, seq_len) - sequence of item IDs
        """
        batch_size, seq_len = item_seq.size()
        
        # Item + positional embeddings
        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=item_seq.device))
        x = item_emb + pos_emb.unsqueeze(0)
        
        # Self-attention layers
        for layer in self.attention_layers:
            x = layer(x)
        
        # Predict next item
        logits = self.output_layer(x)  # (batch_size, seq_len, n_items)
        return logits

# Training
model = SASRec(n_items=10000, hidden_dim=128, num_layers=2)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# item_seq: (batch_size, seq_len), next_items: (batch_size, seq_len)
logits = model(item_seq)
loss = criterion(logits.view(-1, logits.size(-1)), next_items.view(-1))
```

**When to Use**:
- Session-based recommendations (e-commerce, video streaming)
- User has clear sequential behavior (playlist, reading list)
- Need to capture short-term interests

**Real-World**: YouTube (sequential viewing), Spotify (playlist continuation)

---

### 4.2 Multi-Task Learning

**Goal**: Optimize multiple objectives simultaneously (CTR, watch time, conversion)

See [modern-algorithms-reference.md](modern-algorithms-reference.md#27-multi-task-learning-for-recsys-mmoe) for full MMoE implementation.

**Key Architectures**:
- **Hard Parameter Sharing**: Shared bottom layers, task-specific top layers
- **MMoE (Multi-gate Mixture-of-Experts)**: Task-specific gating over shared experts
- **PLE (Progressive Layered Extraction)**: Better handles negative transfer

**Example: CTR + Conversion**

```python
# ESMM: Entire Space Multi-Task Model (handles selection bias)
class ESMM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        # Shared embedding
        self.embedding = nn.Embedding(input_dim, 64)
        
        # CTR tower
        self.ctr_tower = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # CVR tower (conversion rate, only on clicked samples)
        self.cvr_tower = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        emb = self.embedding(x).mean(dim=1)
        
        ctr = self.ctr_tower(emb)  # P(click)
        cvr = self.cvr_tower(emb)  # P(conversion | click)
        
        ctcvr = ctr * cvr  # P(conversion) = P(click) × P(conversion | click)
        
        return ctr, ctcvr
```

**When to Use**:
- Multiple business metrics (engagement, revenue, retention)
- Auxiliary tasks can help main task (e.g., like → click)
- Ranking stage optimization

**Real-World**: YouTube (watch time + likes), Alibaba (click + conversion)

---

### 4.3 Context-Aware Recommendations

**Goal**: Incorporate contextual features (time, location, device, weather)

**Approaches**:
1. **Factorization Machines**: Automatically learn feature crosses
2. **Deep models**: Concatenate context to user/item features
3. **Contextual bandits**: Explore-exploit with context

**Example: Time-aware FM**

```python
# Features: [user_id, item_id, hour_of_day, day_of_week, device_type]
from lightfm import LightFM

# User features: demographics + context
user_features = [
    (user_id, ['age_25-34', 'hour_18', 'device_mobile'])
]

# Item features: categories + popularity
item_features = [
    (item_id, ['category_electronics', 'popularity_high'])
]

model = LightFM(loss='warp', no_components=50)
model.fit(
    interactions,
    user_features=user_features,
    item_features=item_features,
    epochs=10
)
```

**When to Use**:
- Context significantly affects preferences (e.g., morning vs evening content)
- Have rich contextual data
- Need real-time personalization

---

### 4.4 Cold Start Solutions

**New Users**:
1. **Onboarding quiz**: Ask about preferences explicitly
2. **Demographic-based**: Use age/gender/location to initialize
3. **Popularity-based**: Show trending items
4. **Explore-exploit (bandits)**: Quickly learn preferences

**New Items**:
1. **Content-based**: Use item features immediately
2. **Early exploration**: Show to diverse users to gather signal
3. **Transfer learning**: Initialize from similar items

**Hybrid Model**:
```python
def hybrid_recommendation(user_id, is_new_user):
    if is_new_user:
        # Cold start: demographic-based
        demographic_scores = demographic_model.predict(user_id)
        popular_scores = get_trending_items()
        return 0.5 * demographic_scores + 0.5 * popular_scores
    else:
        # Warm start: collaborative filtering
        return collaborative_model.predict(user_id)
```

---

## 5. Evaluation & Metrics

### 5.1 Offline Metrics

**Ranking Metrics** (most important for RecSys):

1. **Precision@K**: Fraction of top-K that are relevant
   ```python
   def precision_at_k(recommended, relevant, k):
       recommended_k = recommended[:k]
       hits = len(set(recommended_k) & set(relevant))
       return hits / k
   ```

2. **Recall@K**: Fraction of relevant items in top-K
   ```python
   def recall_at_k(recommended, relevant, k):
       recommended_k = recommended[:k]
       hits = len(set(recommended_k) & set(relevant))
       return hits / len(relevant)
   ```

3. **NDCG@K (Normalized Discounted Cumulative Gain)**: Position-sensitive metric
   ```python
   import numpy as np
   
   def dcg_at_k(relevance_scores, k):
       """
       relevance_scores: list of relevance scores in recommended order
       """
       relevance = np.array(relevance_scores)[:k]
       gains = 2 ** relevance - 1
       discounts = np.log2(np.arange(2, len(gains) + 2))
       return np.sum(gains / discounts)
   
   def ndcg_at_k(recommended, relevant_scores, k):
       dcg = dcg_at_k(recommended, k)
       ideal_dcg = dcg_at_k(sorted(relevant_scores, reverse=True), k)
       return dcg / ideal_dcg if ideal_dcg > 0 else 0
   ```

4. **MAP (Mean Average Precision)**: Mean of precision at each relevant position
   ```python
   def average_precision(recommended, relevant):
       hits = 0
       sum_precisions = 0
       for i, item in enumerate(recommended):
           if item in relevant:
               hits += 1
               sum_precisions += hits / (i + 1)
       return sum_precisions / len(relevant) if relevant else 0
   
   def mean_average_precision(all_recommendations, all_relevant):
       return np.mean([
           average_precision(rec, rel) 
           for rec, rel in zip(all_recommendations, all_relevant)
       ])
   ```

5. **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant item
   ```python
   def reciprocal_rank(recommended, relevant):
       for i, item in enumerate(recommended):
           if item in relevant:
               return 1 / (i + 1)
       return 0
   ```

**Diversity Metrics**:
- **Intra-list diversity**: How diverse are items in recommendation list?
  ```python
  def intra_list_diversity(recommended_items, item_similarity_matrix):
      n = len(recommended_items)
      diversity = 0
      for i in range(n):
          for j in range(i+1, n):
              diversity += 1 - item_similarity_matrix[recommended_items[i], recommended_items[j]]
      return diversity / (n * (n-1) / 2)
  ```

- **Coverage**: % of items ever recommended
  ```python
  def catalog_coverage(all_recommendations, total_items):
      unique_recommended = set(item for rec in all_recommendations for item in rec)
      return len(unique_recommended) / total_items
  ```

### 5.2 Online Metrics

**Engagement Metrics**:
- **CTR (Click-Through Rate)**: % of impressions that get clicked
- **Watch time / Dwell time**: How long user engages with item
- **Conversion rate**: % of clicks that lead to purchase/signup
- **Return rate**: % of users who come back

**Business Metrics**:
- **Revenue per user**: Direct business impact
- **User retention**: 7-day, 30-day retention
- **Session length**: How long users stay on platform

**Novelty & Serendipity**:
- **Novelty**: Recommend items user doesn't know yet
- **Serendipity**: Recommend surprising but relevant items

### 5.3 A/B Testing for RecSys

**Key Considerations**:
1. **Randomization unit**: User-level (not session-level)
2. **Metrics**: Track engagement, revenue, retention
3. **Duration**: Run for 1-2 weeks (account for novelty effect)
4. **Sample size**: Need large samples for small effect sizes

**Example A/B test**:
```python
# Experiment setup
control_group = random.sample(all_users, n_users // 2)
treatment_group = [u for u in all_users if u not in control_group]

# Control: Old algorithm
for user in control_group:
    recommendations = old_algorithm.recommend(user)
    show_recommendations(user, recommendations)

# Treatment: New algorithm
for user in treatment_group:
    recommendations = new_algorithm.recommend(user)
    show_recommendations(user, recommendations)

# Metrics after 2 weeks
control_ctr = compute_ctr(control_group)
treatment_ctr = compute_ctr(treatment_group)

# Statistical test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(control_ctr, treatment_ctr)
print(f"Improvement: {(treatment_ctr - control_ctr) / control_ctr * 100:.2f}%")
print(f"P-value: {p_value:.4f}")
```

See [experimentation-design-guide.md](experimentation-design-guide.md) for detailed A/B testing methodology.

---

## 6. Production Challenges

### 6.1 Scalability

**Challenge**: Serve personalized recommendations to millions of users in < 100ms

**Solutions**:

1. **Two-stage pipeline**: Retrieval (fast) → Ranking (slow on smaller set)
2. **Approximate Nearest Neighbors**: FAISS, Annoy, ScaNN
3. **Pre-computation**: Compute item embeddings offline
4. **Caching**: Cache popular recommendations
5. **Model compression**: Quantization, pruning, distillation

**Architecture**:
```
User Request
    ↓
[Cache Check] ← (hit) → Return cached recommendations
    ↓ (miss)
[Retrieval: Two-Tower + FAISS] → Top 500 candidates (< 10ms)
    ↓
[Ranking: DCN/DeepFM] → Score top 500 (< 50ms)
    ↓
[Re-ranking: Business rules] → Diversity, freshness (< 20ms)
    ↓
Return Top 20 recommendations
```

### 6.2 Real-Time Updates

**Challenge**: User just watched a video, immediately update recommendations

**Solutions**:

1. **Online learning**: Incremental model updates
2. **Streaming features**: Kafka → Feature store → Model serving
3. **Short-term memory**: Session state (Redis) + long-term model

**Example: Session-based real-time**
```python
# Store session state in Redis
def update_session(user_id, item_id, action):
    redis.lpush(f"session:{user_id}", f"{item_id}:{action}")
    redis.ltrim(f"session:{user_id}", 0, 49)  # Keep last 50 items

# Get recommendations with session context
def recommend_with_session(user_id, n=10):
    # Long-term preferences (from model)
    long_term_recs = model.recommend(user_id, n=100)
    
    # Short-term session (from Redis)
    session_items = redis.lrange(f"session:{user_id}", 0, 9)
    
    # Blend: 70% model, 30% session-based
    # ... blending logic ...
    
    return final_recommendations
```

### 6.3 Popularity Bias

**Problem**: Popular items dominate recommendations, long-tail ignored

**Solutions**:

1. **Debiasing**: Down-weight popular items
   ```python
   def debias_scores(scores, item_popularity):
       # Inverse propensity weighting
       weights = 1 / np.sqrt(item_popularity + 1)
       return scores * weights
   ```

2. **Calibration**: Ensure recommendation distribution matches user's historical distribution
   ```python
   def calibrate_recommendations(user_id, recommendations):
       user_history_distribution = get_user_category_distribution(user_id)
       recommended_distribution = get_category_distribution(recommendations)
       
       # Re-rank to match user's historical distribution
       # ... calibration logic ...
   ```

3. **Exploration**: Explicitly add long-tail items
   ```python
   def add_exploration(recommendations, n_explore=3):
       # Replace bottom 3 with random long-tail items
       long_tail_items = sample_long_tail(n=n_explore)
       recommendations[-n_explore:] = long_tail_items
       return recommendations
   ```

### 6.4 Filter Bubble

**Problem**: Users only see similar content → echo chamber

**Solutions**:

1. **Diversity constraints**: Ensure variety in recommendations
   ```python
   def diversify_recommendations(candidates, item_features, n=10):
       selected = [candidates[0]]  # Start with top item
       
       for _ in range(n - 1):
           # Select item maximizing MMR (Maximal Marginal Relevance)
           best_item = None
           best_score = -float('inf')
           
           for item in candidates:
               if item in selected:
                   continue
               
               relevance = item.score
               diversity = min_similarity(item, selected)
               mmr_score = 0.7 * relevance + 0.3 * diversity
               
               if mmr_score > best_score:
                   best_score = mmr_score
                   best_item = item
           
           selected.append(best_item)
       
       return selected
   ```

2. **Serendipity**: Occasionally show surprising items
3. **User control**: Let users adjust "comfort zone" vs "explore"

### 6.5 Fairness

**Problem**: Algorithmic bias against certain groups (creators, users)

**Solutions**:

1. **Provider fairness**: Ensure fair exposure for all creators
   ```python
   def fair_ranking(recommendations, creator_ids, exposure_targets):
       # Ensure each creator gets fair share of impressions
       # ... fairness-aware ranking ...
   ```

2. **User fairness**: Check performance across demographic groups
   ```python
   # Evaluate separately by demographic
   for demographic_group in ['age_18-25', 'age_25-35', 'age_35+']:
       users = get_users_by_demographic(demographic_group)
       group_ndcg = compute_ndcg(users, recommendations)
       print(f"{demographic_group}: NDCG = {group_ndcg:.3f}")
   ```

3. **Debiasing training data**: Re-weight under-represented groups

---

## 7. Architecture Patterns

### 7.1 Three-Stage Pipeline

**Stage 1: Candidate Generation (Retrieval)**
- **Goal**: Reduce millions of items to hundreds
- **Algorithms**: Two-Tower, ALS, Item-Item CF
- **Latency**: < 10ms
- **Recall-focused**: Get all potentially relevant items

**Stage 2: Ranking**
- **Goal**: Rank hundreds of candidates precisely
- **Algorithms**: DCN, DeepFM, NCF, Gradient Boosting
- **Latency**: < 50ms
- **Precision-focused**: Accurate ordering

**Stage 3: Re-ranking**
- **Goal**: Apply business rules, diversity, freshness
- **Algorithms**: MMR (Maximal Marginal Relevance), rule-based
- **Latency**: < 20ms
- **Business logic**: De-duplication, legal compliance, ads insertion

### 7.2 Feature Store

**Purpose**: Centralized storage for features (online + offline)

**Architecture**:
```
Batch Data (Hive/S3)          Stream Data (Kafka)
        ↓                              ↓
    [Offline Features]         [Real-time Features]
        ↓                              ↓
    [Feature Store: Feast/Tecton]
        ↓                              ↓
    [Offline Store]            [Online Store]
    (for training)             (for serving: Redis/DynamoDB)
```

**Example: Feast**
```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Define features
user_features = store.get_online_features(
    features=[
        "user_profile:age",
        "user_profile:gender",
        "user_stats:avg_watch_time_7d",
        "user_stats:num_clicks_24h",
    ],
    entity_rows=[{"user_id": user_id}]
).to_dict()

item_features = store.get_online_features(
    features=[
        "item_content:category",
        "item_stats:view_count_7d",
        "item_stats:ctr_7d",
    ],
    entity_rows=[{"item_id": item_id}]
).to_dict()
```

### 7.3 Model Serving

**Options**:

1. **Batch Serving**: Pre-compute recommendations daily
   - **Pros**: Simple, low latency
   - **Cons**: Not personalized to latest behavior

2. **Real-time Serving**: Compute on-demand
   - **Pros**: Fresh, personalized
   - **Cons**: Higher latency, cost

3. **Hybrid**: Batch (retrieval) + Real-time (ranking)
   - **Best of both worlds**

**Serving Infrastructure**:
```python
# TorchServe or TensorFlow Serving
# API endpoint
@app.post("/recommend")
def recommend(user_id: int, context: dict):
    # Get user features
    user_features = feature_store.get_online_features(user_id)
    
    # Retrieval
    candidates = retrieval_model.retrieve(user_features, k=500)
    
    # Ranking
    ranked = ranking_model.rank(user_features, candidates)
    
    # Re-ranking
    final = rerank_with_business_rules(ranked, context)
    
    return final[:20]
```

---

## 8. Case Studies

### 8.1 YouTube Recommendations

**System** (2016 paper: "Deep Neural Networks for YouTube Recommendations"):

**Architecture**:
1. **Candidate Generation**:
   - Two-Tower model (video embeddings + user embeddings)
   - Train on watch history (implicit feedback)
   - Retrieve top 500 from millions using ANN

2. **Ranking**:
   - Deep neural network (800+ features)
   - Predict watch time (not just CTR)
   - Features: user demographics, video metadata, context (time, device)

**Key Insights**:
- **Optimize for watch time, not CTR**: Prevents clickbait
- **Example age**: Recent videos get exploration boost
- **Negative sampling**: Important for implicit feedback
- **Serving**: Pre-compute video embeddings, only compute user embedding at request time

**Scale**: 1 billion+ users, 400 hours/min uploaded

---

### 8.2 Netflix Recommendations

**System** (2010s):

**Algorithms**:
- **Personalized ranking**: Row-based UI (each row is a different algorithm)
  - "Because you watched X"
  - "Trending now"
  - "Top picks for you"
  - "New releases"

- **Thumbnail personalization**: Different artwork for different users
- **Multi-armed bandits**: Explore which artwork/titles work best

**Key Insights**:
- **A/B test everything**: Rigorous experimentation culture
- **Business metrics**: Focus on retention, not just engagement
- **Offline ≠ Online**: Offline metrics don't always correlate with business metrics
- **Diversity**: Ensure variety across rows

**Scale**: 200M+ subscribers, 80% watch time from recommendations

---

### 8.3 TikTok For You Page

**System** (estimated from public info):

**Architecture**:
1. **Candidate Generation**: Two-Tower (user interests + video embeddings)
2. **Ranking**: Multi-task learning (watch time, likes, shares, comments)
3. **Calibration**: Cold start (new users see diverse content)

**Key Features**:
- **Video embeddings**: From visual content (CV models) + audio + text
- **User embeddings**: From watch history, interactions, device info
- **Context**: Time of day, trending topics

**Key Insights**:
- **Optimize for multiple signals**: Not just watch time, but likes, shares, comments
- **Short feedback loop**: Videos are short → fast learning
- **Exploration heavy**: New users see diverse content to learn preferences quickly
- **Virality**: Algorithm amplifies videos with high early engagement

**Scale**: 1 billion+ users, average 95 minutes/day

---

### 8.4 Amazon Product Recommendations

**System** ("Customers who bought this also bought"):

**Algorithms**:
- **Item-item collaborative filtering**: Core algorithm (2003 paper)
  - Scalable: Pre-compute item-item similarities
  - Explainable: "Customers who bought X also bought Y"

**Architecture**:
1. **Batch**: Compute item-item similarities daily
2. **Online**: Lookup pre-computed similarities

**Key Insights**:
- **Item-item > User-user**: More stable, easier to explain
- **Implicit feedback**: Purchases, views, add-to-cart
- **Context matters**: "Frequently bought together" for complementary items
- **Long tail**: Recommendations help long-tail products

**Scale**: 12M+ products

---

## Summary: Decision Tree for RecSys

```
Start
  ↓
Do you have user-item interactions?
  ├─ No → Content-based filtering
  └─ Yes → Continue
      ↓
Data size?
  ├─ Small (< 100K) → Collaborative Filtering (Item-Item CF)
  ├─ Medium (100K - 10M) → Matrix Factorization (ALS, BPR)
  └─ Large (> 10M) → Neural RecSys (Two-Tower + Ranking model)
      ↓
Need real-time?
  ├─ No → Batch recommendations (daily pre-compute)
  └─ Yes → Two-stage pipeline (Retrieval + Ranking)
      ↓
Multiple objectives?
  ├─ No → Single-task model
  └─ Yes → Multi-task learning (MMoE, ESMM)
      ↓
Cold start critical?
  ├─ No → Pure collaborative filtering
  └─ Yes → Hybrid (Collaborative + Content-based)
```

---

## Production Checklist

**Before Launch**:
- [ ] A/B test with 5% traffic for 1 week
- [ ] Monitor latency (p50, p95, p99 < SLA)
- [ ] Check diversity metrics (coverage > 10% of catalog)
- [ ] Evaluate across demographic groups (fairness)
- [ ] Load test (10x expected traffic)
- [ ] Set up monitoring dashboards (CTR, latency, errors)
- [ ] Document feature dependencies
- [ ] Prepare rollback plan

**After Launch**:
- [ ] Daily metric monitoring (CTR, engagement, revenue)
- [ ] Weekly A/B test reviews
- [ ] Monthly model retraining
- [ ] Quarterly cold start analysis
- [ ] Bi-annual algorithm refresh (explore new methods)

---

## Further Reading

**Papers**:
- YouTube (2016): "Deep Neural Networks for YouTube Recommendations"
- Netflix (2009): "The Netflix Prize"
- Amazon (2003): "Item-Based Collaborative Filtering Recommendation Algorithms"
- Wide & Deep (2016): Google
- NCF (2017): "Neural Collaborative Filtering"
- SASRec (2018): "Self-Attentive Sequential Recommendation"
- MMoE (2018): "Modeling Task Relationships in Multi-task Learning"

**Books**:
- "Recommender Systems: The Textbook" by Charu Aggarwal
- "Practical Recommender Systems" by Kim Falk

**Tools**:
- **Surprise**: Python library for classical algorithms
- **Implicit**: Fast ALS/BPR implementations
- **LightFM**: Hybrid recommender systems
- **TensorFlow Recommenders**: End-to-end RecSys with TF
- **FAISS**: Billion-scale similarity search (Meta)
- **Feast**: Open-source feature store

---

**Last Updated**: 2026-01-27  
**Maintainer**: Data Science Team
