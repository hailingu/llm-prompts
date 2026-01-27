# Classic Data Science Algorithms

This document serves as a reference guide for classic, proven algorithms that every data scientist should know. These algorithms have stood the test of time and remain highly effective for many problems.

---

## Table of Contents

1. [Supervised Learning](#supervised-learning)
   - [Regression](#regression)
   - [Classification](#classification)
   - [Ensemble Methods](#ensemble-methods)
2. [Unsupervised Learning](#unsupervised-learning)
3. [Dimensionality Reduction](#dimensionality-reduction)
4. [Time Series](#time-series)
5. [Deep Learning (Classical Architectures)](#deep-learning-classical-architectures)
6. [Reinforcement Learning (Classical)](#reinforcement-learning-classical)

---

## Supervised Learning

### Regression

#### 1. Linear Regression (OLS)
**When to use**:
- Linear relationship between features and target
- Interpretability is critical
- Quick baseline for regression problems

**Pros**:
- Simple and interpretable
- Fast training and prediction
- Well-understood statistical properties

**Cons**:
- Assumes linearity
- Sensitive to outliers
- Cannot capture complex non-linear relationships

**Python Example**:
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**When NOT to use**:
- Highly non-linear relationships
- High-dimensional data with multicollinearity

---

#### 2. Ridge Regression (L2 Regularization)
**When to use**:
- Multicollinearity in features
- Prevent overfitting
- Many correlated features

**Pros**:
- Handles multicollinearity well
- Reduces overfitting
- More stable than OLS

**Cons**:
- Doesn't perform feature selection (keeps all features)
- Requires tuning regularization parameter

**Python Example**:
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha controls regularization strength
model.fit(X_train, y_train)
```

**Regularization parameter**:
- `alpha = 0`: Same as OLS
- `alpha > 0`: Stronger regularization

---

#### 3. Lasso Regression (L1 Regularization)
**When to use**:
- Feature selection needed
- Many irrelevant features
- Sparse models preferred

**Pros**:
- Performs automatic feature selection (sets coefficients to 0)
- Handles high-dimensional data well
- Produces interpretable models

**Cons**:
- Can struggle with correlated features
- May arbitrarily select one from correlated group

**Python Example**:
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# Check selected features
selected_features = X.columns[model.coef_ != 0]
```

---

#### 4. Polynomial Regression
**When to use**:
- Non-linear relationships
- Curved patterns in data

**Pros**:
- Captures non-linear relationships
- Still interpretable

**Cons**:
- Prone to overfitting with high degrees
- Requires careful degree selection

**Python Example**:
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
model.fit(X_train, y_train)
```

---

### Classification

#### 5. Logistic Regression
**When to use**:
- Binary or multi-class classification
- Need probability estimates
- Interpretability is important
- Baseline classification model

**Pros**:
- Simple and fast
- Provides probability outputs
- Well-calibrated probabilities
- Interpretable coefficients

**Cons**:
- Assumes linear decision boundary
- Cannot handle complex non-linear patterns

**Python Example**:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Get probabilities
y_proba = model.predict_proba(X_test)[:, 1]
```

**Regularization options**:
- `penalty='l2'`: Ridge (default)
- `penalty='l1'`: Lasso (with solver='saga')
- `C`: Inverse regularization strength (smaller = stronger regularization)

---

#### 6. Decision Tree
**When to use**:
- Non-linear decision boundaries
- Feature interactions important
- Interpretability needed (small trees)
- Mixed feature types (numerical + categorical)

**Pros**:
- Highly interpretable (can visualize)
- Handles non-linear relationships
- No feature scaling required
- Handles missing values naturally

**Cons**:
- Prone to overfitting
- Unstable (small data changes → big tree changes)
- Biased towards features with more levels

**Python Example**:
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(
    max_depth=5,           # Limit depth to prevent overfitting
    min_samples_split=20,  # Minimum samples to split node
    min_samples_leaf=10,   # Minimum samples in leaf
    random_state=42
)
model.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=feature_names, class_names=['0', '1'], filled=True)
plt.savefig('decision_tree.png')
```

**Hyperparameters to tune**:
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split
- `min_samples_leaf`: Minimum samples in leaf
- `max_features`: Features to consider for split

---

#### 7. Random Forest
**When to use**:
- Tabular data (structured)
- Default choice for many problems
- Feature importance needed
- Robust model required

**Pros**:
- Excellent out-of-box performance
- Reduces overfitting vs single tree
- Provides feature importance
- Handles missing values
- Minimal hyperparameter tuning needed

**Cons**:
- Less interpretable than single tree
- Slower training than single tree
- Can be memory-intensive

**Python Example**:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Max depth per tree
    min_samples_split=20,
    max_features='sqrt',   # Features per split
    n_jobs=-1,             # Use all CPU cores
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

**Key hyperparameters**:
- `n_estimators`: More is better (100-500 typical)
- `max_depth`: Control overfitting (5-15 typical)
- `max_features`: 'sqrt' for classification, 'log2' or None for options

---

#### 8. Gradient Boosting (XGBoost, LightGBM, CatBoost)
**When to use**:
- **XGBoost**: General-purpose, proven track record
- **LightGBM**: Large datasets (>10k rows), faster training
- **CatBoost**: Many categorical features, less tuning needed

**Pros**:
- State-of-the-art performance on tabular data
- Feature importance
- Handles missing values
- Built-in regularization

**Cons**:
- Requires careful hyperparameter tuning
- Prone to overfitting without proper tuning
- Slower than Random Forest

**XGBoost Example**:
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,             # Minimum loss reduction
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=100
)
```

**LightGBM Example** (faster for large data):
```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.01,
    num_leaves=31,         # LightGBM-specific
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
```

**CatBoost Example** (handles categorical features):
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.01,
    cat_features=categorical_feature_indices,  # Automatic handling
    random_state=42,
    verbose=100
)
model.fit(X_train, y_train)
```

**Comparison**:
| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Speed | Medium | Fast | Medium-Fast |
| Categorical handling | Manual encoding | Manual encoding | Automatic |
| Default performance | Good | Good | Excellent |
| Tuning required | High | Medium | Low |

---

#### 9. Support Vector Machine (SVM)
**When to use**:
- Small to medium datasets (< 10k samples)
- High-dimensional data
- Clear margin of separation exists

**Pros**:
- Effective in high dimensions
- Memory efficient (uses support vectors only)
- Versatile (different kernel functions)

**Cons**:
- Slow on large datasets (> 10k)
- Requires feature scaling
- Not well-suited for large datasets
- Difficult to interpret

**Python Example**:
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# SVM requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(
    kernel='rbf',          # 'linear', 'poly', 'rbf', 'sigmoid'
    C=1.0,                 # Regularization parameter
    gamma='scale',         # Kernel coefficient
    probability=True,      # Enable probability estimates
    random_state=42
)
model.fit(X_train_scaled, y_train)
```

**Kernel selection**:
- `linear`: Linear decision boundary
- `rbf`: Non-linear (Gaussian), most common
- `poly`: Polynomial boundary
- `sigmoid`: Similar to neural networks

---

#### 10. Naive Bayes
**When to use**:
- Text classification (spam detection, sentiment analysis)
- High-dimensional data
- Need fast training and prediction
- Baseline probabilistic model

**Pros**:
- Very fast training and prediction
- Works well with high-dimensional data
- Requires little training data
- Naturally handles multi-class

**Cons**:
- Assumes feature independence (often violated)
- Can be outperformed by more complex models

**Python Example**:
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Gaussian NB: Continuous features
model = GaussianNB()
model.fit(X_train, y_train)

# Multinomial NB: Count data (text classification)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(texts_train)

model = MultinomialNB(alpha=1.0)  # Laplace smoothing
model.fit(X_train_counts, y_train)
```

**Variants**:
- **GaussianNB**: Continuous features (assumes Gaussian distribution)
- **MultinomialNB**: Count data (word counts in text)
- **BernoulliNB**: Binary features

---

### Ensemble Methods

#### 11. Bagging (Bootstrap Aggregating)
**Concept**: Train multiple models on different random subsets of data

**When to use**:
- Reduce variance of high-variance models (e.g., decision trees)
- Have enough data for sampling
- Want model stability

**Python Example**:
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=50,
    max_samples=0.8,      # 80% of data per bag
    max_features=0.8,     # 80% of features per bag
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
```

---

#### 12. Boosting (AdaBoost)
**Concept**: Sequentially train models, focusing on misclassified samples

**When to use**:
- Improve weak learners
- Have clean data (sensitive to outliers)

**Python Example**:
```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
model.fit(X_train, y_train)
```

---

#### 13. Voting Classifier
**Concept**: Combine predictions from multiple different algorithms

**When to use**:
- Have several good models
- Want to leverage strengths of different algorithms

**Python Example**:
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

model = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('svm', SVC(probability=True))
    ],
    voting='soft',        # 'hard' for majority vote, 'soft' for avg probabilities
    weights=[2, 1, 1]     # Weight models differently
)
model.fit(X_train, y_train)
```

---

## Unsupervised Learning

### Clustering

#### 14. K-Means Clustering
**When to use**:
- Globular, equal-sized clusters expected
- Need fast clustering
- Know approximate number of clusters

**Pros**:
- Simple and fast
- Scales well to large datasets
- Easy to interpret

**Cons**:
- Must specify K (number of clusters)
- Assumes spherical clusters
- Sensitive to initialization
- Sensitive to outliers

**Python Example**:
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method to find optimal K
inertias = []
silhouette_scores = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X, model.labels_))

# Plot and choose K
# Then train final model
model = KMeans(n_clusters=5, random_state=42)
clusters = model.fit_predict(X)
```

---

#### 15. Hierarchical Clustering
**When to use**:
- Don't know number of clusters in advance
- Want dendrogram (cluster hierarchy)
- Small to medium datasets

**Pros**:
- No need to specify number of clusters upfront
- Produces dendrogram (interpretable)
- Deterministic (no random initialization)

**Cons**:
- Slow on large datasets (O(n²) or O(n³))
- Cannot undo merges

**Python Example**:
```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Create dendrogram
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.savefig('dendrogram.png')

# Agglomerative clustering
model = AgglomerativeClustering(n_clusters=5, linkage='ward')
clusters = model.fit_predict(X)
```

**Linkage methods**:
- `ward`: Minimizes variance within clusters (most common)
- `complete`: Maximum distance
- `average`: Average distance
- `single`: Minimum distance

---

#### 16. DBSCAN (Density-Based Clustering)
**When to use**:
- Arbitrary-shaped clusters
- Outlier detection needed
- Don't know number of clusters

**Pros**:
- Finds arbitrary-shaped clusters
- Identifies outliers as noise
- No need to specify number of clusters

**Cons**:
- Requires tuning epsilon and min_samples
- Struggles with varying density clusters

**Python Example**:
```python
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.5,              # Maximum distance between points
    min_samples=5,        # Minimum points to form cluster
    n_jobs=-1
)
clusters = model.fit_predict(X)

# -1 indicates outliers/noise
outliers = X[clusters == -1]
```

---

## Dimensionality Reduction

#### 17. Principal Component Analysis (PCA)
**When to use**:
- Reduce dimensionality
- Visualize high-dimensional data
- Remove correlated features
- Speed up algorithms

**Pros**:
- Linear transformation (fast)
- Preserves variance
- Removes multicollinearity

**Cons**:
- Assumes linearity
- Principal components not interpretable
- Sensitive to scale (requires normalization)

**Python Example**:
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# PCA requires scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Explained variance plot
pca_full = PCA()
pca_full.fit(X_scaled)

plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig('pca_variance.png')

# Select n_components to retain 95% variance
pca = PCA(n_components=0.95)  # or explicit number like 10
X_reduced = pca.fit_transform(X_scaled)
```

---

#### 18. t-SNE (t-Distributed Stochastic Neighbor Embedding)
**When to use**:
- Visualize high-dimensional data (2D/3D)
- Explore cluster structure

**Pros**:
- Excellent for visualization
- Preserves local structure well

**Cons**:
- Computationally expensive
- Non-deterministic (different results each run)
- Cannot be applied to new data (no transform)
- Sensitive to hyperparameters

**Python Example**:
```python
from sklearn.manifold import TSNE

model = TSNE(
    n_components=2,        # 2D visualization
    perplexity=30,         # Balance local vs global structure
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
X_embedded = model.fit_transform(X)

# Visualize
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
plt.savefig('tsne_plot.png')
```

**Important**: t-SNE is for visualization only, not feature engineering!

---

#### 19. UMAP (Uniform Manifold Approximation and Projection)
**When to use**:
- Alternative to t-SNE
- Faster than t-SNE
- Better preservation of global structure

**Pros**:
- Faster than t-SNE
- Can transform new data
- Preserves both local and global structure

**Cons**:
- Requires installation (`pip install umap-learn`)

**Python Example**:
```python
import umap

model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)
X_embedded = model.fit_transform(X)
```

---

## Time Series

#### 20. ARIMA (AutoRegressive Integrated Moving Average)
**When to use**:
- Univariate time series forecasting
- Stationary or near-stationary series
- Need interpretable model

**Pros**:
- Well-established statistical method
- Interpretable parameters
- Handles seasonality (SARIMA)

**Cons**:
- Requires stationarity
- Manual parameter tuning (p, d, q)
- Struggles with complex patterns

**Python Example**:
```python
from statsmodels.tsa.arima.model import ARIMA

# Auto ARIMA to find optimal (p,d,q)
from pmdarima import auto_arima

auto_model = auto_arima(
    y_train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True
)
print(auto_model.summary())

# Fit ARIMA with found parameters
model = ARIMA(y_train, order=(2, 1, 2))  # (p, d, q)
fitted_model = model.fit()

# Forecast
forecast = fitted_model.forecast(steps=30)
```

---

#### 21. Prophet (Facebook)
**When to use**:
- Business time series with strong seasonality
- Missing data or outliers
- Multiple seasonality patterns (daily + yearly)

**Pros**:
- Handles missing data and outliers
- Automatic detection of seasonality
- Easy to use
- Incorporates holidays

**Cons**:
- May overfit on small datasets
- Less flexible than custom models

**Python Example**:
```python
from prophet import Prophet
import pandas as pd

# Data must have 'ds' (date) and 'y' (value) columns
df = pd.DataFrame({'ds': dates, 'y': values})

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Visualize
model.plot(forecast)
model.plot_components(forecast)
```

---

## Deep Learning (Classical Architectures)

#### 22. Multi-Layer Perceptron (MLP) / Feedforward Neural Network
**When to use**:
- Non-linear relationships
- Tabular data (after trying gradient boosting)
- Large datasets

**Pros**:
- Can learn complex non-linear patterns
- Universal function approximator

**Cons**:
- Requires large datasets
- Prone to overfitting
- Requires careful hyperparameter tuning
- Less interpretable

**PyTorch Example**:
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

model = MLP(input_dim=50, hidden_dims=[128, 64, 32], output_dim=1)
```

---

#### 23. Convolutional Neural Network (CNN)
**When to use**:
- Image classification
- Spatial patterns in data
- Local feature extraction

**Architecture highlights**:
- Convolutional layers extract features
- Pooling layers downsample
- Fully connected layers for classification

---

#### 24. Recurrent Neural Network (RNN / LSTM / GRU)
**When to use**:
- Sequential data (time series, text)
- Variable-length input
- Temporal dependencies

**Variants**:
- **RNN**: Basic, suffers from vanishing gradients
- **LSTM**: Solves vanishing gradient with gates
- **GRU**: Simpler than LSTM, often similar performance

---
Reinforcement Learning (Classical)

### 25. Q-Learning
**When to use**:
- Discrete state and action spaces
- Model-free learning
- Tabular environments (small state spaces)

**Pros**:
- Simple and proven
- Guarantees convergence to optimal policy
- No need for environment model

**Cons**:
- Only works for discrete spaces
- Doesn't scale to large state spaces
- Requires extensive exploration

**Python Example**:
```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

# Training loop
agent = QLearningAgent(n_states=100, n_actions=4)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
```

**Use Cases**:
- Grid world navigation
- Simple game playing
- Basic robotics control

---

### 26. SARSA (State-Action-Reward-State-Action)
**When to use**:
- On-policy learning needed
- Safe exploration important
- Similar to Q-Learning but more conservative

**Key Difference from Q-Learning**:
- Q-Learning: Off-policy (learns optimal policy while following exploratory policy)
- SARSA: On-policy (learns the policy it's following)

**Python Example**:
```python
class SARSAAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, next_action):
        # SARSA update rule (uses actual next action, not max)
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
```

**When to prefer over Q-Learning**:
- Safety-critical applications (SARSA is more conservative)
- When actual behavior policy matters

---

### 27. Deep Q-Network (DQN)
**When to use**:
- Large/continuous state spaces (images, etc.)
- Discrete actions
- Need to scale Q-learning

**Key Innovation**: Use neural network to approximate Q-function

**Python Example (using PyTorch)**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = deque(maxlen=10000)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_network.network[-1].out_features - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute Q(s,a)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target: r + γ * max_a' Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.MSELoss()(q_values.squeeze(), targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Training loop
agent = DQNAgent(state_dim=4, action_dim=2)

for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    for step in range(500):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, done)
        agent.train(batch_size=64)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Update target network every N episodes
    if episode % 10 == 0:
        agent.update_target_network()
    
    # Decay epsilon
    agent.epsilon = max(0.01, agent.epsilon * 0.995)
```

**Key Techniques**:
1. **Experience Replay**: Store transitions, sample randomly to break correlation
2. **Target Network**: Separate network for stable targets
3. **Epsilon Decay**: Gradually shift from exploration to exploitation

**Improvements (2015+)**:
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Sample important transitions more

---

### 28. Policy Gradient Methods (REINFORCE)
**When to use**:
- Continuous action spaces
- Stochastic policies needed
- Direct policy optimization

**Key Idea**: Directly optimize policy parameters to maximize expected reward

**Python Example**:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
    
    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def train(self, log_probs, rewards):
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        # Normalize returns (optional but helps)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Policy gradient loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training loop
agent = REINFORCEAgent(state_dim=4, action_dim=2)

for episode in range(1000):
    state = env.reset()
    log_probs = []
    rewards = []
    
    done = False
    while not done:
        action, log_prob = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        
        state = next_state
    
    # Train after episode
    agent.train(log_probs, rewards)
```

**Pros**:
- Works with continuous action spaces
- Can learn stochastic policies
- Simple to implement

**Cons**:
- High variance
- Sample inefficient
- Sensitive to hyperparameters

---

### 29. Actor-Critic
**When to use**:
- Need lower variance than REINFORCE
- Continuous or discrete actions
- Balance between policy gradient and value-based

**Key Idea**: Combine policy (actor) and value function (critic)

**Python Example**:
```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Actor (policy network)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value network)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.ac = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma = gamma
    
    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = self.ac(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), state_value
    
    def train(self, log_prob, state_value, reward, next_state_value, done):
        # Compute TD error (advantage)
        td_target = reward + (1 - done) * self.gamma * next_state_value
        td_error = td_target - state_value
        
        # Actor loss (policy gradient with advantage)
        actor_loss = -log_prob * td_error.detach()
        
        # Critic loss (MSE)
        critic_loss = td_error.pow(2)
        
        # Total loss
        loss = actor_loss + critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training loop (online)
agent = ActorCriticAgent(state_dim=4, action_dim=2)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action, log_prob, state_value = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Get next state value
        with torch.no_grad():
            _, next_state_value = agent.ac(torch.FloatTensor(next_state).unsqueeze(0))
        
        # Train online
        agent.train(log_prob, state_value, reward, next_state_value, done)
        
        state = next_state
```

**Advantages over REINFORCE**:
- Lower variance (uses value baseline)
- Can train online (step-by-step)
- More sample efficient

---

### 30. Deep Deterministic Policy Gradient (DDPG)
**When to use**:
- Continuous action spaces (robotics, control)
- Deterministic policies
- Model-free learning

**Key Idea**: Actor-Critic for continuous actions + DQN techniques

**Python Example**:
```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=1e-3, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau  # Soft update parameter
        self.replay_buffer = deque(maxlen=100000)
    
    def choose_action(self, state, noise=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        # Add exploration noise
        action += noise * np.random.randn(len(action))
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
    
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

**Use Cases**:
- Robotic control
- Continuous control tasks
- Autonomous driving

---

### 31. Advantage Actor-Critic (A2C) / A3C
**Architecture**: Synchronous (A2C) or Asynchronous (A3C) Actor-Critic

**When to use**:
- Parallel training
- Need faster learning
- Multi-core CPUs available (A3C)

**Key Innovation**: Multiple parallel workers collecting experience

**Python Example (A2C - simpler than A3C)**:
```python
import torch.multiprocessing as mp

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, n_workers=4):
        self.shared_model = ActorCritic(state_dim, action_dim).share_memory()
        self.optimizer = optim.Adam(self.shared_model.parameters(), lr=lr)
        self.gamma = gamma
        self.n_workers = n_workers
    
    def worker(self, worker_id, env_fn):
        env = env_fn()
        state = env.reset()
        
        while True:
            # Collect trajectory
            states, actions, rewards, log_probs, values = [], [], [], [], []
            
            for _ in range(20):  # Collect 20 steps
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, value = self.shared_model(state_tensor)
                
                dist = Categorical(action_probs)
                action = dist.sample()
                
                next_state, reward, done, _ = env.step(action.item())
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(dist.log_prob(action))
                values.append(value)
                
                state = next_state
                if done:
                    state = env.reset()
            
            # Compute returns and advantages
            returns = self.compute_returns(rewards, values)
            
            # Update shared model
            self.update(log_probs, values, returns)
    
    def compute_returns(self, rewards, values):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)
    
    def update(self, log_probs, values, returns):
        log_probs = torch.stack(log_probs)
        values = torch.cat(values)
        
        advantages = returns - values.detach()
        
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, env_fn):
        # Start multiple worker processes
        processes = []
        for worker_id in range(self.n_workers):
            p = mp.Process(target=self.worker, args=(worker_id, env_fn))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
```

**A3C vs A2C**:
- **A3C**: Asynchronous updates (workers update independently)
- **A2C**: Synchronous updates (wait for all workers, then update)
- **A2C is simpler and often performs similarly**

---

## 
## Algorithm Selection Flowchart

```
Start
  |
  ├─ Tabular Data?
  │    ├─ Yes → Try: Gradient Boosting (XGBoost/LightGBM) → Random Forest → MLP
  │    └─ No → Go to Data Type
  │
  ├─ Image Data?
  │    └─ Yes → CNN (ResNet, EfficientNet)
  │
  ├─ Text Data?
  │    └─ Yes → Transformer (BERT, RoBERTa) or Naive Bayes (simple)
  │
  ├─ Time Series?
  │    └─ Yes → ARIMA → Prophet → LSTM
  │
  ├─ Clustering?
  │    └─ Yes → K-Means → DBSCAN → Hierarchical
  │
  └─ Recommendation?
       └─ Yes → Collaborative Filtering → Matrix Factorization → Neural RecSys
```

---

## 30. Collaborative Filtering (User-Based & Item-Based)

### When to Use
- User-item interaction data (ratings, clicks, purchases)
- Need personalized recommendations
- Sufficient historical data (typically > 100 interactions per user)
- Can tolerate cold start issues

### Pros
- **Simple and interpretable**: Easy to explain recommendations
- **No domain knowledge needed**: Works with interaction data only
- **Flexible**: Works with explicit (ratings) or implicit (clicks) feedback
- **Proven track record**: Used by Amazon, Netflix early systems

### Cons
- **Cold start problem**: Cannot recommend for new users/items
- **Scalability**: Computing all pairwise similarities is O(n²)
- **Sparsity**: Struggles with sparse interaction matrices
- **Popularity bias**: Tends to recommend popular items

### Python Implementation

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-based Collaborative Filtering
class UserBasedCF:
    def __init__(self, k=10):
        self.k = k  # Number of similar users
        
    def fit(self, user_item_matrix):
        """
        user_item_matrix: shape (n_users, n_items)
        """
        # Compute user-user similarity (cosine)
        self.user_similarity = cosine_similarity(user_item_matrix)
        self.user_item_matrix = user_item_matrix
        
    def predict(self, user_id, item_id):
        """Predict rating for user_id on item_id"""
        # Find k most similar users who rated this item
        similar_users = np.argsort(self.user_similarity[user_id])[::-1][1:self.k+1]
        
        # Filter users who rated this item
        rated_mask = self.user_item_matrix[similar_users, item_id] > 0
        similar_users = similar_users[rated_mask]
        
        if len(similar_users) == 0:
            return self.user_item_matrix[:, item_id].mean()
        
        # Weighted average of ratings
        similarities = self.user_similarity[user_id, similar_users]
        ratings = self.user_item_matrix[similar_users, item_id]
        
        return np.dot(similarities, ratings) / similarities.sum()
    
    def recommend(self, user_id, n=10):
        """Recommend top-n items for user_id"""
        # Get items user hasn't interacted with
        unrated_items = np.where(self.user_item_matrix[user_id] == 0)[0]
        
        # Predict ratings for all unrated items
        predictions = [(item_id, self.predict(user_id, item_id)) 
                       for item_id in unrated_items]
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

# Item-based Collaborative Filtering
class ItemBasedCF:
    def __init__(self, k=10):
        self.k = k
        
    def fit(self, user_item_matrix):
        # Compute item-item similarity
        self.item_similarity = cosine_similarity(user_item_matrix.T)
        self.user_item_matrix = user_item_matrix
        
    def recommend(self, user_id, n=10):
        """Recommend based on similar items to what user liked"""
        # Get items user has interacted with
        user_items = np.where(self.user_item_matrix[user_id] > 0)[0]
        
        # Aggregate similarity scores for unrated items
        unrated_items = np.where(self.user_item_matrix[user_id] == 0)[0]
        scores = {}
        
        for item_id in unrated_items:
            # Sum similarity to all items user liked
            similar_scores = self.item_similarity[item_id, user_items]
            scores[item_id] = similar_scores.sum()
        
        # Sort and return top-n
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return top_items[:n]

# Example usage
user_item_matrix = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 2],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 0],
    [0, 3, 4, 0, 1],
])

# User-based CF
ubcf = UserBasedCF(k=2)
ubcf.fit(user_item_matrix)
recommendations = ubcf.recommend(user_id=0, n=3)
print(f"User-based recommendations: {recommendations}")

# Item-based CF
ibcf = ItemBasedCF(k=3)
ibcf.fit(user_item_matrix)
recommendations = ibcf.recommend(user_id=0, n=3)
print(f"Item-based recommendations: {recommendations}")
```

### When NOT to Use
- New users/items dominate (cold start)
- Need real-time recommendations at massive scale
- Interaction matrix extremely sparse (< 0.1% density)
- Need to incorporate content features (use hybrid or content-based)

### Real-World Applications
- **Amazon** (Item-item CF): "Customers who bought this also bought..."
- **Netflix** (early system): Movie recommendations based on similar users
- **Last.fm**: Music recommendations based on listening history

### Production Considerations
- **Scalability**: Use approximate nearest neighbors (Annoy, FAISS) for large-scale
- **Incremental updates**: Recompute similarities periodically, not per interaction
- **Hybrid approach**: Combine with content-based for cold start handling

---

## 31. Matrix Factorization (SVD, ALS, NMF)

### When to Use
- Large-scale recommendation systems (millions of users/items)
- Sparse user-item interaction matrices
- Need low-dimensional embeddings for users and items
- Collaborative filtering with implicit or explicit feedback

### Pros
- **Scalable**: Efficient for large datasets (Netflix Prize winner)
- **Handles sparsity well**: Learns latent factors
- **Embeddings**: User/item vectors useful for downstream tasks
- **Flexible loss functions**: Can optimize for ranking, rating prediction, etc.

### Cons
- **Cold start**: Cannot embed new users/items without retraining
- **Hyperparameter sensitive**: Number of factors, regularization critical
- **Not interpretable**: Latent factors hard to explain
- **Linear interactions**: Cannot capture complex nonlinear patterns

### Python Implementation

```python
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

# Alternating Least Squares (ALS) - for implicit feedback
def als_recommendations():
    # Create sparse user-item matrix (implicit feedback: 1 = interaction, 0 = no interaction)
    user_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    item_ids = np.array([0, 1, 1, 2, 0, 3, 2, 3])
    interactions = np.ones(len(user_ids))
    
    user_item_matrix = coo_matrix(
        (interactions, (user_ids, item_ids)),
        shape=(4, 4)
    ).tocsr()
    
    # Train ALS model
    model = AlternatingLeastSquares(
        factors=50,          # Latent dimensions
        regularization=0.01, # L2 regularization
        iterations=15,
        calculate_training_loss=True
    )
    
    model.fit(user_item_matrix)
    
    # Get recommendations for user 0
    user_id = 0
    recommendations = model.recommend(
        userid=user_id,
        user_items=user_item_matrix[user_id],
        N=5,
        filter_already_liked_items=True
    )
    
    print(f"ALS Recommendations for user {user_id}:")
    for item_id, score in recommendations:
        print(f"  Item {item_id}: score {score:.3f}")
    
    # Get similar items
    item_id = 1
    similar_items = model.similar_items(item_id, N=3)
    print(f"\nItems similar to item {item_id}:")
    for similar_id, score in similar_items:
        print(f"  Item {similar_id}: similarity {score:.3f}")
    
    return model

# Bayesian Personalized Ranking (BPR) - ranking-based MF
def bpr_recommendations():
    # Same data format
    user_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    item_ids = np.array([0, 1, 1, 2, 0, 3, 2, 3])
    interactions = np.ones(len(user_ids))
    
    user_item_matrix = coo_matrix(
        (interactions, (user_ids, item_ids)),
        shape=(4, 4)
    ).tocsr()
    
    # Train BPR model (optimizes for ranking)
    model = BayesianPersonalizedRanking(
        factors=50,
        learning_rate=0.01,
        regularization=0.01,
        iterations=100
    )
    
    model.fit(user_item_matrix)
    
    # Get recommendations
    user_id = 0
    recommendations = model.recommend(
        userid=user_id,
        user_items=user_item_matrix[user_id],
        N=5
    )
    
    print(f"\nBPR Recommendations for user {user_id}:")
    for item_id, score in recommendations:
        print(f"  Item {item_id}: score {score:.3f}")
    
    return model

# SVD-based Matrix Factorization (explicit ratings)
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

def svd_recommendations():
    # Explicit ratings data
    data = [
        ('user1', 'item1', 5.0),
        ('user1', 'item2', 3.0),
        ('user2', 'item1', 4.0),
        ('user2', 'item3', 2.0),
        ('user3', 'item2', 5.0),
        ('user3', 'item3', 4.0),
    ]
    
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(
        pd.DataFrame(data, columns=['user', 'item', 'rating']),
        reader
    )
    
    # Train SVD
    svd = SVD(
        n_factors=50,        # Latent dimensions
        n_epochs=20,
        lr_all=0.005,        # Learning rate
        reg_all=0.02         # Regularization
    )
    
    # Cross-validation
    results = cross_validate(svd, dataset, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    
    # Train on full dataset
    trainset = dataset.build_full_trainset()
    svd.fit(trainset)
    
    # Predict rating
    prediction = svd.predict('user1', 'item3')
    print(f"\nPredicted rating for user1 on item3: {prediction.est:.2f}")
    
    return svd

# Run examples
als_model = als_recommendations()
bpr_model = bpr_recommendations()
svd_model = svd_recommendations()
```

### Algorithms Comparison

| Algorithm | Feedback Type | Loss Function | Best For |
|-----------|---------------|---------------|----------|
| **SVD** | Explicit (ratings) | MSE | Rating prediction |
| **ALS** | Implicit (clicks) | Weighted MSE | Large-scale implicit feedback |
| **BPR** | Implicit | Pairwise ranking | Ranking optimization |
| **NMF** | Non-negative | KL divergence | Interpretable factors |

### When NOT to Use
- Need to incorporate item/user features → Use Factorization Machines or neural models
- Cold start is critical → Use content-based or hybrid
- Need real-time updates → Matrix factorization requires batch retraining

### Real-World Applications
- **Netflix Prize**: SVD-based models won (2009)
- **Spotify**: ALS for playlist continuation
- **Pinterest**: Modified ALS for pin recommendations
- **YouTube**: ALS for candidate generation (early system)

### Production Tips
- **Implicit library**: Fast C++ implementation for ALS/BPR
- **Negative sampling**: For implicit feedback, sample negative items
- **Confidence weighting**: Weight interactions by frequency (e.g., # of clicks)
- **Embeddings**: Extract user/item vectors for downstream tasks (search, clustering)

---

## 32. Factorization Machines (FM)

### When to Use
- High-dimensional sparse feature interactions (e.g., user + item + context features)
- CTR prediction (click-through rate)
- Need to model pairwise feature interactions efficiently
- Combining collaborative filtering with content features

### Pros
- **Feature interactions**: Automatically learns pairwise feature crosses
- **Handles sparsity**: Works well with sparse categorical features
- **General framework**: Subsumes matrix factorization, polynomial regression
- **Efficient**: Linear complexity in number of features

### Cons
- **Only 2nd order**: Cannot capture higher-order interactions (use FFM, DeepFM)
- **Linear base**: Still has linear component limitations
- **Hyperparameter tuning**: Factor size matters

### Python Implementation

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import lightfm
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

# Factorization Machine for CTR prediction
class FactorizationMachine:
    def __init__(self, n_factors=10, lr=0.01, reg=0.01, epochs=20):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w0 = 0.0  # Bias
        self.w = np.zeros(n_features)  # Linear weights
        self.V = np.random.normal(0, 0.01, (n_features, self.n_factors))  # Factor matrix
        
        # SGD training
        for epoch in range(self.epochs):
            for i in range(n_samples):
                x = X[i].toarray().flatten() if hasattr(X[i], 'toarray') else X[i]
                y_pred = self._predict_instance(x)
                error = y[i] - y_pred
                
                # Update bias
                self.w0 += self.lr * error
                
                # Update linear weights
                self.w += self.lr * (error * x - self.reg * self.w)
                
                # Update factor matrix
                for f in range(self.n_factors):
                    sum_vx = np.dot(self.V[:, f], x)
                    for j in range(n_features):
                        if x[j] != 0:
                            self.V[j, f] += self.lr * (error * x[j] * (sum_vx - self.V[j, f] * x[j]) - self.reg * self.V[j, f])
            
            if epoch % 5 == 0:
                loss = np.mean([(y[i] - self._predict_instance(X[i]))**2 for i in range(min(1000, n_samples))])
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def _predict_instance(self, x):
        # Bias + linear term
        pred = self.w0 + np.dot(self.w, x)
        
        # Interaction term: ½ Σf [(Σi vi,f xi)² - Σi vi,f² xi²]
        for f in range(self.n_factors):
            sum_vx = np.dot(self.V[:, f], x)
            sum_vx_sq = np.dot(self.V[:, f]**2, x**2)
            pred += 0.5 * (sum_vx**2 - sum_vx_sq)
        
        return pred
    
    def predict(self, X):
        return np.array([self._predict_instance(x.toarray().flatten() if hasattr(x, 'toarray') else x) for x in X])

# Example: CTR prediction with sparse features
from scipy.sparse import csr_matrix

# Simulated data: user_id, item_id, category, day_of_week → click (0/1)
# One-hot encoded features
n_samples = 1000
n_users = 100
n_items = 50
n_categories = 10

user_ids = np.random.randint(0, n_users, n_samples)
item_ids = np.random.randint(0, n_items, n_samples)
categories = np.random.randint(0, n_categories, n_samples)
clicks = (np.random.rand(n_samples) > 0.7).astype(int)

# Create sparse feature matrix (one-hot encoding)
n_features = n_users + n_items + n_categories
X = np.zeros((n_samples, n_features))
for i in range(n_samples):
    X[i, user_ids[i]] = 1  # User feature
    X[i, n_users + item_ids[i]] = 1  # Item feature
    X[i, n_users + n_items + categories[i]] = 1  # Category feature

X_sparse = csr_matrix(X)

# Train FM
fm = FactorizationMachine(n_factors=10, lr=0.01, reg=0.001, epochs=20)
fm.fit(X_sparse, clicks)

# Predict
predictions = fm.predict(X_sparse[:10])
print("\nPredictions (first 10):", predictions)
print("Actual (first 10):", clicks[:10])

# Using LightFM library (better production implementation)
print("\n=== LightFM Implementation ===")
from lightfm.data import Dataset

dataset = Dataset()
dataset.fit(users=range(n_users), items=range(n_items))

(interactions, weights) = dataset.build_interactions(
    [(user_ids[i], item_ids[i]) for i in range(n_samples) if clicks[i] == 1]
)

model = LightFM(
    no_components=10,  # Number of latent factors
    loss='warp',       # WARP loss for ranking
    learning_rate=0.05,
    random_state=42
)

model.fit(interactions, epochs=10, num_threads=2)

# Evaluation
precision = precision_at_k(model, interactions, k=5).mean()
print(f"Precision@5: {precision:.3f}")
```

### When NOT to Use
- Need deep feature interactions → Use DeepFM or xDeepFM
- Text/image features → Use neural embeddings + FM hybrid
- Extremely high-dimensional → Consider feature selection first

### Real-World Applications
- **CTR prediction**: Online advertising (Google, Facebook)
- **Recommendation**: Combining user/item features with context
- **Ranking**: Search results ranking with query + document features

---

## Key Takeaways

1. **Start Simple**: Always begin with logistic regression or random forest as baseline
2. **Gradient Boosting for Tabular**: XGBoost/LightGBM are kings of tabular data
3. **Deep Learning for Unstructured**: Use CNNs for images, Transformers for text
4. **Collaborative Filtering for RecSys**: Start with item-item CF, scale with Matrix Factorization
5. **Feature Engineering > Algorithm**: Better features beat better algorithms
6. **Cross-Validation Always**: Never trust a model without proper validation

---

**Last Updated**: 2026-01-27  
**Version**: 1.1 (added recommender systems algorithms)  
**Maintainer**: Data Science Team
