# Experimentation Design Guide

**Purpose**: Rigorous experimental methodology for data science — A/B testing, causal inference, and statistical best practices.

**Target Audience**: Research leads, data scientists, evaluators

**Last Updated**: 2026-01-27

---

## Table of Contents

1. [A/B Testing Fundamentals](#ab-testing-fundamentals)
2. [Statistical Power Analysis](#statistical-power-analysis)
3. [Advanced Experimental Designs](#advanced-experimental-designs)
4. [Causal Inference Methods](#causal-inference-methods)
5. [Multiple Testing Corrections](#multiple-testing-corrections)
6. [Real-World Considerations](#real-world-considerations)
7. [Tools and Frameworks](#tools-and-frameworks)

---

## A/B Testing Fundamentals

### What is A/B Testing?

**Definition**: Controlled experiment comparing two or more variants to determine which performs better on a key metric.

**Key Principles**:
- **Randomization**: Users randomly assigned to treatment/control
- **Controlled**: Only one variable changes between groups
- **Statistical**: Decisions based on statistical significance, not intuition

### When to Use A/B Testing

✅ **Use A/B testing when**:
- You want to measure **causal impact** (not just correlation)
- You can randomly assign users to variants
- You have enough traffic for statistical power
- The metric is measurable and business-relevant

❌ **Don't use A/B testing when**:
- Sample size too small (< 1000 per variant)
- Can't randomize (use quasi-experimental design)
- Effect is expected to be tiny (< 0.5% lift)
- Cost of running test > expected value

---

### A/B Test Workflow

```
1. Define hypothesis
   "Changing button color from blue to green will increase CTR by 5%"
   
2. Choose metrics
   Primary: Click-through rate (CTR)
   Secondary: Conversion rate, revenue per user
   Guardrail: Page load time, bounce rate
   
3. Calculate sample size
   Power analysis: Need N users per variant
   
4. Randomize and collect data
   50/50 split, run for T days
   
5. Analyze results
   Statistical significance test
   Confidence intervals
   
6. Make decision
   Ship, iterate, or abandon
```

---

### Example: Button Color A/B Test

```python
import numpy as np
from scipy import stats

# Scenario: Button color change
# Control (blue): 10,000 users, 1,000 clicks (10% CTR)
# Treatment (green): 10,000 users, 1,100 clicks (11% CTR)

n_control = 10000
clicks_control = 1000
ctr_control = clicks_control / n_control

n_treatment = 10000
clicks_treatment = 1100
ctr_treatment = clicks_treatment / n_treatment

# Z-test for proportions
z_stat, p_value = stats.proportions_ztest(
    [clicks_treatment, clicks_control],
    [n_treatment, n_control]
)

print(f"Control CTR: {ctr_control:.2%}")
print(f"Treatment CTR: {ctr_treatment:.2%}")
print(f"Lift: {(ctr_treatment - ctr_control) / ctr_control:.2%}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Statistically significant! Ship treatment.")
else:
    print("❌ Not significant. Need more data or abandon.")
```

---

### Common A/B Testing Mistakes

#### Mistake 1: Peeking (Sequential Testing Fallacy)
**Problem**: Checking results before planned end, stopping early when "significant"

**Why it's wrong**: Increases false positive rate (Type I error)

**Solution**: 
- Pre-commit to sample size
- Use sequential testing methods (see Advanced Designs)
- Never stop early just because p < 0.05

```python
# ❌ WRONG: Peeking
for day in range(1, 30):
    p_value = run_test(day)
    if p_value < 0.05:
        print(f"Stop on day {day}! Significant!")
        break  # FALSE POSITIVE RISK!

# ✅ CORRECT: Pre-commit
planned_days = 14
run_test_for_n_days(planned_days)
p_value = run_final_test()
```

#### Mistake 2: Not Accounting for Multiple Testing
**Problem**: Testing 20 variants, finding 1 "significant" (expected by chance)

**Solution**: Bonferroni correction or False Discovery Rate (FDR)

```python
from statsmodels.stats.multitest import multipletests

# Test 10 variants
p_values = [0.01, 0.04, 0.03, 0.001, 0.08, 0.15, 0.20, 0.50, 0.60, 0.70]

# Bonferroni correction
rejected, corrected_p, _, _ = multipletests(p_values, method='bonferroni', alpha=0.05)
print(f"Significant after correction: {sum(rejected)}")

# FDR (Benjamini-Hochberg) - less conservative
rejected_fdr, corrected_p_fdr, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
print(f"Significant with FDR: {sum(rejected_fdr)}")
```

#### Mistake 3: Ignoring Sample Ratio Mismatch (SRM)
**Problem**: Expected 50/50 split, got 52/48 split (data quality issue!)

**Detection**:
```python
from scipy.stats import chi2_contingency

# Expected 50/50, observed 5200/4800
observed = [5200, 4800]
expected = [5000, 5000]

chi2, p_value, _, _ = chi2_contingency([observed, expected])

if p_value < 0.001:  # Very low threshold
    print("⚠️ Sample Ratio Mismatch! Check randomization.")
```

---

## Statistical Power Analysis

### The Four Quantities

Every A/B test involves 4 quantities (knowing 3, we can compute the 4th):

1. **Effect Size (δ)**: Minimum detectable difference (e.g., 5% lift in CTR)
2. **Sample Size (n)**: Users per variant
3. **Significance Level (α)**: False positive rate (typically 0.05)
4. **Statistical Power (1-β)**: True positive rate (typically 0.80)

### Why Power Analysis Matters

**Underpowered test**: Wastes time, can't detect real effects  
**Overpowered test**: Wastes traffic, delays other experiments

---

### Power Analysis for Proportions (CTR, Conversion)

```python
from statsmodels.stats.power import zt_ind_solve_power

# Scenario: CTR improvement test
# Current CTR: 10%
# Want to detect: 1% absolute lift (10% → 11%, relative lift of 10%)
# Significance: 0.05
# Power: 0.80

baseline_rate = 0.10
target_rate = 0.11
effect_size = (target_rate - baseline_rate) / np.sqrt(baseline_rate * (1 - baseline_rate))

n_per_variant = zt_ind_solve_power(
    effect_size=effect_size,
    alpha=0.05,
    power=0.80,
    alternative='larger'
)

print(f"Need {n_per_variant:.0f} users per variant")
print(f"Total sample size: {2 * n_per_variant:.0f}")

# Example output: Need 15,730 users per variant (31,460 total)
```

### Power Analysis for Continuous Metrics (Revenue, Time)

```python
from statsmodels.stats.power import tt_ind_solve_power

# Scenario: Revenue per user improvement
# Current avg revenue: $50, std: $20
# Want to detect: $2.5 increase (5% lift)
# Significance: 0.05
# Power: 0.80

mean_baseline = 50
std_baseline = 20
mde = 2.5  # Minimum Detectable Effect
effect_size = mde / std_baseline  # Cohen's d

n_per_variant = tt_ind_solve_power(
    effect_size=effect_size,
    alpha=0.05,
    power=0.80,
    alternative='larger'
)

print(f"Need {n_per_variant:.0f} users per variant")
```

---

### Minimum Detectable Effect (MDE)

If you have **fixed sample size**, calculate the **smallest effect you can detect**:

```python
# Given: 10,000 users per variant
# What's the MDE?

n_per_variant = 10000
alpha = 0.05
power = 0.80

effect_size = zt_ind_solve_power(
    nobs1=n_per_variant,
    alpha=alpha,
    power=power,
    alternative='larger'
)

baseline_rate = 0.10
mde = effect_size * np.sqrt(baseline_rate * (1 - baseline_rate))
target_rate = baseline_rate + mde

print(f"Baseline: {baseline_rate:.2%}")
print(f"MDE: {mde:.2%} absolute ({mde/baseline_rate:.1%} relative)")
print(f"Target: {target_rate:.2%}")

# Output: Can detect ~0.87% absolute lift (8.7% relative lift)
```

---

### Practical Rules of Thumb

| Metric Type   | Baseline   | Realistic MDE                | Sample Size (per variant)   |
| ------------- | ---------- | ---------------------------- | --------------------------- |
| ------------- | ---------- | ---------------              | --------------------------- |
| CTR           | 10%        | 0.5% absolute (5% relative)  | ~60,000                     |
| CTR           | 10%        | 1% absolute (10% relative)   | ~15,000                     |
| Conversion    | 5%         | 0.25% absolute (5% relative) | ~120,000                    |
| Revenue       | $50 ± $20  | $1 (2% relative)             | ~6,300                      |

**Takeaway**: Small lifts need HUGE sample sizes!

---

## Advanced Experimental Designs

### 1. Multi-Armed Bandits (MAB)

**Problem with A/B testing**: Wastes 50% of traffic on inferior variant

**MAB solution**: Dynamically allocate more traffic to better-performing variant

#### Thompson Sampling

```python
import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.successes = np.ones(n_arms)  # Beta prior: α=1
        self.failures = np.ones(n_arms)   # Beta prior: β=1
    
    def select_arm(self):
        # Sample from posterior Beta(α, β) for each arm
        samples = np.random.beta(self.successes, self.failures)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        if reward:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

# Usage
bandit = ThompsonSampling(n_arms=2)  # A vs B

for i in range(10000):
    arm = bandit.select_arm()
    reward = simulate_reward(arm)  # Your reward function
    bandit.update(arm, reward)

# Arm 0 selected 7,843 times (78%)
# Arm 1 selected 2,157 times (22%)
# Automatically allocated more traffic to winner!
```

#### When to Use MAB vs A/B

| Use A/B Testing                   | Use Multi-Armed Bandit                |
| --------------------------------- | ------------------------------------- |
| -----------------                 | ------------------------              |
| Need precise effect size estimate | Care more about total reward          |
| Low traffic (< 10K/day)           | High traffic (> 100K/day)             |
| Long-term decision (ship or not)  | Short-term optimization (daily deals) |
| Single experiment                 | Continuous optimization               |

---

### 2. Sequential Testing (Early Stopping)

**Problem**: A/B tests run for fixed duration, even when result is obvious

**Solution**: Sequential Probability Ratio Test (SPRT) or mSPRT

```python
# Simple sequential testing with alpha spending
from scipy.stats import norm

def sequential_test(clicks_a, impressions_a, clicks_b, impressions_b, alpha=0.05):
    """
    Sequential test with O'Brien-Fleming alpha spending.
    """
    p_a = clicks_a / impressions_a
    p_b = clicks_b / impressions_b
    
    # Pooled proportion
    p_pool = (clicks_a + clicks_b) / (impressions_a + impressions_b)
    
    # Z-statistic
    se = np.sqrt(p_pool * (1 - p_pool) * (1/impressions_a + 1/impressions_b))
    z_stat = (p_b - p_a) / se
    
    # O'Brien-Fleming boundary (simplified)
    # Adjust alpha based on information fraction
    n_current = impressions_a + impressions_b
    n_planned = 20000  # Planned total sample size
    info_frac = n_current / n_planned
    
    # Alpha spending: more conservative early, normal threshold near end
    alpha_spent = alpha * info_frac
    z_boundary = norm.ppf(1 - alpha_spent / 2)
    
    if abs(z_stat) > z_boundary:
        return "STOP", z_stat, alpha_spent
    else:
        return "CONTINUE", z_stat, alpha_spent

# Check every 1000 impressions
for n in range(1000, 20000, 1000):
    decision, z, alpha_t = sequential_test(clicks_a, n//2, clicks_b, n//2)
    if decision == "STOP":
        print(f"Stop at n={n}: z={z:.2f}, adjusted α={alpha_t:.4f}")
        break
```

---

### 3. Stratified Experiments

**Problem**: User segments behave differently, but we test on aggregate

**Solution**: Stratify by segment, test within each

```python
import pandas as pd

# Example: Mobile vs Desktop users
df = pd.DataFrame({
    'user_id': range(10000),
    'platform': np.random.choice(['mobile', 'desktop'], 10000, p=[0.7, 0.3]),
    'variant': np.random.choice(['A', 'B'], 10000),
    'converted': np.random.binomial(1, 0.1, 10000)
})

# Stratified analysis
for platform in ['mobile', 'desktop']:
    df_strata = df[df['platform'] == platform]
    
    conversion_a = df_strata[df_strata['variant'] == 'A']['converted'].mean()
    conversion_b = df_strata[df_strata['variant'] == 'B']['converted'].mean()
    
    print(f"{platform.capitalize()}:")
    print(f"  Variant A: {conversion_a:.2%}")
    print(f"  Variant B: {conversion_b:.2%}")
    print(f"  Lift: {(conversion_b - conversion_a) / conversion_a:.1%}")
    print()

# Combine using weighted average
# This accounts for different segment sizes
```

---

### 4. Switchback Experiments

**Problem**: Network effects or supply-side constraints (e.g., ride-sharing surge pricing)

**Solution**: Switch entire market between A and B over time

```python
# Example: Testing surge pricing algorithm
# Can't run A/B on same city (drivers would see inconsistent prices)
# Solution: Switch NYC between algorithm A and B every hour

import pandas as pd

df = pd.DataFrame({
    'hour': range(24),
    'algorithm': ['A', 'B'] * 12,  # Alternate every hour
    'driver_earnings': np.random.normal(100, 20, 24),
    'rider_wait_time': np.random.normal(5, 2, 24)
})

# Compare
earnings_a = df[df['algorithm'] == 'A']['driver_earnings'].mean()
earnings_b = df[df['algorithm'] == 'B']['driver_earnings'].mean()

print(f"Algorithm A earnings: ${earnings_a:.2f}")
print(f"Algorithm B earnings: ${earnings_b:.2f}")

# Statistical test accounting for time-series structure
from statsmodels.stats.weightstats import ttest_ind

t_stat, p_value, _ = ttest_ind(
    df[df['algorithm'] == 'A']['driver_earnings'],
    df[df['algorithm'] == 'B']['driver_earnings']
)
```

---

## Causal Inference Methods

### When A/B Testing is Not Possible

Sometimes you **cannot randomize**:
- Historical data (can't go back in time)
- Ethical concerns (can't randomize medical treatment)
- Technical limitations (can't randomize certain users)

**Solution**: Quasi-experimental designs that approximate causality

---

### 1. Propensity Score Matching (PSM)

**Idea**: Match treated and control units with similar characteristics

**Steps**:
1. Estimate propensity score: P(treatment | X)
2. Match treated to control with similar propensity
3. Compare outcomes

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# Example: Email campaign (users self-selected)
df = pd.DataFrame({
    'user_id': range(1000),
    'age': np.random.normal(35, 10, 1000),
    'tenure_days': np.random.exponential(365, 1000),
    'previous_purchases': np.random.poisson(5, 1000),
    'received_email': np.random.binomial(1, 0.3, 1000),  # Non-random!
    'purchased': np.random.binomial(1, 0.1, 1000)
})

# Step 1: Estimate propensity score
X = df[['age', 'tenure_days', 'previous_purchases']]
y = df['received_email']

propensity_model = LogisticRegression()
propensity_model.fit(X, y)
df['propensity'] = propensity_model.predict_proba(X)[:, 1]

# Step 2: Match treated to control
treated = df[df['received_email'] == 1]
control = df[df['received_email'] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity']].values)

distances, indices = nn.kneighbors(treated[['propensity']].values)

# Step 3: Compare outcomes
matched_control = control.iloc[indices.flatten()]
treatment_effect = treated['purchased'].mean() - matched_control['purchased'].mean()

print(f"Average Treatment Effect (ATE): {treatment_effect:.2%}")
```

**Limitations**:
- Assumes **no unobserved confounders** (strong assumption!)
- Only works if overlap in propensity scores

---

### 2. Difference-in-Differences (DiD)

**Idea**: Compare change in treatment group vs change in control group

**Setup**: Treatment applied at time T
- Group A: Treated (gets intervention at T)
- Group B: Control (never treated)

```python
import pandas as pd

# Example: New feature launched in California but not Texas
df = pd.DataFrame({
    'state': ['CA'] * 200 + ['TX'] * 200,
    'time': ['pre'] * 100 + ['post'] * 100 + ['pre'] * 100 + ['post'] * 100,
    'revenue': (
        np.random.normal(100, 10, 100).tolist() +  # CA pre
        np.random.normal(115, 10, 100).tolist() +  # CA post (+15)
        np.random.normal(100, 10, 100).tolist() +  # TX pre
        np.random.normal(105, 10, 100).tolist()    # TX post (+5)
    )
})

# Calculate DiD
ca_pre = df[(df['state'] == 'CA') & (df['time'] == 'pre')]['revenue'].mean()
ca_post = df[(df['state'] == 'CA') & (df['time'] == 'post')]['revenue'].mean()
tx_pre = df[(df['state'] == 'TX') & (df['time'] == 'pre')]['revenue'].mean()
tx_post = df[(df['state'] == 'TX') & (df['time'] == 'post')]['revenue'].mean()

ca_change = ca_post - ca_pre
tx_change = tx_post - tx_pre
did_estimate = ca_change - tx_change

print(f"CA change: ${ca_change:.2f}")
print(f"TX change: ${tx_change:.2f}")
print(f"DiD estimate: ${did_estimate:.2f}")
# Output: DiD estimate: $10 (true effect = 15 - 5 = 10)

# Regression approach (more robust)
import statsmodels.formula.api as smf

df['treated'] = (df['state'] == 'CA').astype(int)
df['post'] = (df['time'] == 'post').astype(int)

model = smf.ols('revenue ~ treated * post', data=df).fit()
print(model.summary())
# Coefficient on treated:post is DiD estimate
```

**Assumptions**:
- **Parallel trends**: Without treatment, groups would have same trend
- **No spillover**: Treatment doesn't affect control group

---

### 3. Regression Discontinuity Design (RDD)

**Idea**: Treatment assigned based on cutoff (e.g., credit score > 700)

**Setup**:
- Running variable X (e.g., credit score)
- Cutoff C (e.g., 700)
- Treatment if X > C

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example: Loan approval based on credit score cutoff
np.random.seed(42)
credit_score = np.random.uniform(600, 800, 1000)
cutoff = 700

# Treatment: approved if score > 700
approved = (credit_score > cutoff).astype(int)

# Outcome: income (treatment causes +$5k income boost)
income = 50 + 0.05 * credit_score + approved * 5 + np.random.normal(0, 5, 1000)

# RDD: Compare just above vs just below cutoff
bandwidth = 20
near_cutoff = np.abs(credit_score - cutoff) < bandwidth

income_above = income[(credit_score > cutoff) & near_cutoff].mean()
income_below = income[(credit_score <= cutoff) & near_cutoff].mean()

treatment_effect = income_above - income_below

print(f"Income just above cutoff: ${income_above:.2f}k")
print(f"Income just below cutoff: ${income_below:.2f}k")
print(f"Treatment effect: ${treatment_effect:.2f}k")

# Visualize
plt.scatter(credit_score, income, alpha=0.3, s=10)
plt.axvline(cutoff, color='red', linestyle='--', label='Cutoff')
plt.xlabel('Credit Score')
plt.ylabel('Income ($k)')
plt.title('Regression Discontinuity Design')
plt.legend()
plt.show()
```

---

### 4. Synthetic Control

**Idea**: Create "synthetic" control by weighted combination of untreated units

**Use Case**: Policy evaluation when only one treated unit (e.g., California carbon tax)

```python
# Example: Feature launched in one city (NYC), need to estimate impact
# Use weighted combination of other cities as synthetic NYC

import pandas as pd
from sklearn.linear_model import LinearRegression

# Pre-period data (before treatment)
cities = ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
pre_period_revenue = {
    'NYC': [100, 102, 105, 107],
    'LA': [80, 81, 83, 84],
    'Chicago': [60, 61, 62, 63],
    'Houston': [70, 71, 72, 73],
    'Phoenix': [50, 51, 52, 53]
}

# Find weights to match NYC in pre-period
X = np.array([pre_period_revenue[city] for city in cities[1:]]).T  # Control cities
y = np.array(pre_period_revenue['NYC'])  # Treated city

# Constrained regression (weights sum to 1, non-negative)
from scipy.optimize import minimize

def objective(weights):
    synthetic = X @ weights
    return np.sum((y - synthetic) ** 2)

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1)] for _ in range(4)]

result = minimize(objective, x0=[0.25]*4, bounds=bounds, constraints=constraints)
weights = result.x

print("Synthetic NYC weights:")
for city, weight in zip(cities[1:], weights):
    print(f"  {city}: {weight:.2f}")

# Post-period: Compare actual NYC to synthetic NYC
post_period_revenue = {
    'NYC': [110, 115, 120],  # Treatment effect!
    'LA': [85, 86, 87],
    'Chicago': [64, 65, 66],
    'Houston': [74, 75, 76],
    'Phoenix': [54, 55, 56]
}

synthetic_nyc = np.array([post_period_revenue[city] for city in cities[1:]]).T @ weights
actual_nyc = np.array(post_period_revenue['NYC'])

treatment_effect = actual_nyc - synthetic_nyc
print(f"\\nTreatment effect per period: {treatment_effect}")
```

---

### 5. Instrumental Variables (IV)

**Idea**: Use instrument Z that affects treatment but not outcome (except through treatment)

**Requirements for valid instrument**:
1. **Relevance**: Z affects treatment T
2. **Exclusion**: Z only affects outcome Y through T
3. **Exogeneity**: Z is not correlated with confounders

```python
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS

# Example: Effect of education on income
# Problem: Ability (unobserved) affects both education and income
# Instrument: Distance to college (affects education, not income directly)

df = pd.DataFrame({
    'income': np.random.normal(50, 20, 1000),
    'education_years': np.random.normal(14, 3, 1000),
    'distance_to_college': np.random.exponential(10, 1000)
})

# Stage 1: Regress education on instrument
stage1 = sm.OLS(
    df['education_years'],
    sm.add_constant(df['distance_to_college'])
).fit()

# Stage 2: Use predicted education in outcome regression
df['education_pred'] = stage1.fittedvalues

stage2 = sm.OLS(
    df['income'],
    sm.add_constant(df['education_pred'])
).fit()

print(stage2.summary())
```

---

### Causal Inference with DoWhy

```python
import dowhy
from dowhy import CausalModel

# Example: Email campaign effect on purchase
df = pd.DataFrame({
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.normal(50, 20, 1000),
    'email_sent': np.random.binomial(1, 0.5, 1000),
    'purchased': np.random.binomial(1, 0.1, 1000)
})

# Define causal model
model = CausalModel(
    data=df,
    treatment='email_sent',
    outcome='purchased',
    common_causes=['age', 'income']
)

# Identify causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate effect
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)

print(f"Causal effect: {estimate.value:.3f}")

# Refute (sensitivity analysis)
refutation = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause"
)
print(refutation)
```

---

## Multiple Testing Corrections

### The Problem

**Scenario**: Test 20 features, significance level α = 0.05

**Expected false positives**: 20 × 0.05 = 1 "significant" result by chance!

**Solution**: Adjust significance threshold

---

### Bonferroni Correction

**Method**: Divide α by number of tests

```python
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Test 20 features
n_tests = 20
p_values = []

for i in range(n_tests):
    group_a = np.random.normal(0, 1, 100)
    group_b = np.random.normal(0.2, 1, 100)  # Only first 5 have real effect
    _, p = ttest_ind(group_a, group_b)
    p_values.append(p)

# Without correction
significant_uncorrected = sum(p < 0.05 for p in p_values)
print(f"Significant (uncorrected): {significant_uncorrected}")

# Bonferroni correction
rejected, p_corrected, _, _ = multipletests(p_values, method='bonferroni', alpha=0.05)
print(f"Significant (Bonferroni): {sum(rejected)}")
```

**Pros**: Simple, controls Family-Wise Error Rate (FWER)  
**Cons**: Too conservative for many tests

---

### False Discovery Rate (FDR) - Benjamini-Hochberg

**Method**: Control proportion of false positives among all rejections

```python
# FDR (Benjamini-Hochberg) - less conservative
rejected_fdr, p_corrected_fdr, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
print(f"Significant (FDR): {sum(rejected_fdr)}")
```

**Pros**: More power than Bonferroni  
**Cons**: Allows some false positives (controlled proportion)

---

### When to Use Which

| Method            | Use When                              |
| ----------------- | ------------------------------------- |
| --------          | ----------                            |
| **Bonferroni**    | Few tests (< 10), need strong control |
| **FDR (BH)**      | Many tests (10-1000), exploration     |
| **No correction** | Single pre-planned test               |

---

## Real-World Considerations

### 1. Network Effects

**Problem**: User A's treatment affects User B's outcome

**Example**: Ride-sharing surge pricing — if we raise prices for 50% of riders, drivers migrate to those areas, affecting other 50%

**Solutions**:
- Cluster randomization (randomize by geography/time)
- Switchback experiments
- Use methods that account for interference

---

### 2. Novelty Effects

**Problem**: Users excited by new feature, but effect fades

**Solution**:
- Run experiments for longer (weeks, not days)
- Monitor metrics over time
- Compare early vs late periods

```python
# Detect novelty effect
df['week'] = df['day'] // 7

for week in df['week'].unique():
    lift_week = calculate_lift(df[df['week'] == week])
    print(f"Week {week}: {lift_week:.1%} lift")

# If lift decreases over time → novelty effect
```

---

### 3. Seasonality

**Problem**: Metrics vary by day of week, season, holidays

**Solution**:
- Run experiments for full weeks (not Mon-Wed)
- Stratify by time period
- Use matched pairs (compare same days)

---

### 4. Sample Size Estimation with Historical Variance

```python
# Use historical data to estimate variance for power analysis
historical_revenue = df['revenue'].values

mean_revenue = historical_revenue.mean()
std_revenue = historical_revenue.std()

# Power analysis
mde = 0.05 * mean_revenue  # 5% lift
effect_size = mde / std_revenue

n_per_variant = tt_ind_solve_power(effect_size=effect_size, alpha=0.05, power=0.80)
print(f"Need {n_per_variant:.0f} users per variant")
```

---

## Tools and Frameworks

### Python Libraries

| Library              | Use Case                               |
| -------------------- | -------------------------------------- |
| ---------            | ----------                             |
| **scipy.stats**      | Basic statistical tests                |
| **statsmodels**      | Power analysis, regression             |
| **dowhy**            | Causal inference                       |
| **causalml (Uber)**  | Uplift modeling, heterogeneous effects |
| **PyMC**             | Bayesian A/B testing                   |
| **Weights & Biases** | Experiment tracking                    |

### Commercial Platforms

| Platform            | Features                        |
| ------------------- | ------------------------------- |
| ----------          | ----------                      |
| **Optimizely**      | Full-service A/B testing        |
| **Google Optimize** | Free, integrates with Analytics |
| **LaunchDarkly**    | Feature flags + experiments     |
| **VWO**             | A/B testing + personalization   |

---

## Summary: Experimentation Checklist

### Before Starting Experiment
- [ ] Hypothesis clearly defined
- [ ] Primary metric chosen (only one!)
- [ ] Secondary and guardrail metrics defined
- [ ] Sample size calculated (power analysis)
- [ ] Randomization strategy defined
- [ ] Duration estimated
- [ ] Pre-commit to stopping rule (no peeking!)

### During Experiment
- [ ] Monitor sample ratio mismatch (SRM)
- [ ] Check data quality daily
- [ ] Don't peek at results (unless using sequential testing)

### After Experiment
- [ ] Statistical significance test
- [ ] Confidence intervals reported
- [ ] Check for novelty effects
- [ ] Segment analysis (any subgroups different?)
- [ ] Decision: ship, iterate, or abandon

### For Causal Inference (Non-Randomized)
- [ ] Identify potential confounders
- [ ] Check overlap/common support
- [ ] Validate assumptions (parallel trends, exclusion restriction, etc.)
- [ ] Sensitivity analysis / robustness checks
- [ ] Acknowledge limitations

---

**Key Takeaway**: **Randomization is gold standard**. When not possible, use causal inference methods with caution and always validate assumptions.

---

**Last Updated**: 2026-01-27  
**Version**: 1.0  
**Maintainer**: Data Science Team
