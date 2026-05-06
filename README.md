# -Decision-Tree-and-Random-Forest-Regression
A complete beginner-friendly walkthrough of Decision Tree and Random Forest Regression using Python and Scikit-learn. This project uses the California Housing Dataset to predict house prices.


## 📚 Table of Contents

- [What is Machine Learning?](#what-is-machine-learning)
- [What is Regression?](#what-is-regression)
- [Decision Tree Regression](#decision-tree-regression)
- [Parts of a Decision Tree](#parts-of-a-decision-tree)
- [The Dataset](#the-dataset)
- [Data Preparation](#data-preparation)
- [Building the Decision Tree Model](#building-the-decision-tree-model)
- [Decision Tree Results](#decision-tree-results)
- [Feature Importance - Decision Tree](#feature-importance---decision-tree)
- [Random Forest Regression](#random-forest-regression)
- [Building the Random Forest Model](#building-the-random-forest-model)
- [Random Forest Results](#random-forest-results)
- [Feature Importance - Random Forest](#feature-importance---random-forest)
- [Model Comparison](#model-comparison)
- [Key Concepts Glossary](#key-concepts-glossary)

---

## 🤔 What is Machine Learning?

Machine Learning is teaching a computer to learn from examples — just like how humans learn.

For example:
- A child sees many cats and dogs → eventually they can tell the difference
- A computer sees many houses and their prices → eventually it can predict the price of a new house

The computer **studies old data** and **learns patterns** from it.

---

## 🎯 What is Regression?

Regression means **predicting a number.**

| Question | Answer (a number) |
|---|---|
| How much will this house cost? | $250,000 |
| What will the temperature be tomorrow? | 32°C |
| How much will this person earn? | $60,000/year |

---

## 🌳 Decision Tree Regression

A Decision Tree Regression predicts a continuous numeric value by learning a series of yes/no questions that split the data based on the most important features.

Think of it like a **game of 20 Questions**:

```
Is house size > 150 sqm?
├── YES → Is income > $50,000?
│         ├── YES → Predicted Price: $450,000
│         └── NO  → Predicted Price: $300,000
└── NO  → Is house age < 10 years?
          ├── YES → Predicted Price: $200,000
          └── NO  → Predicted Price: $120,000
```

---

## 🧩 Parts of a Decision Tree

| Part | What it is | Simple Meaning |
|---|---|---|
| **Root Node** | The very first question | Where the tree starts — contains ALL the data |
| **Internal Node** | Questions in the middle | Keeps narrowing things down |
| **Leaf Node** | The final answer | A number — the average price of that group |
| **Branch** | Lines connecting questions | The YES or NO path you follow |

### How the Leaf Node Predicts

When a house lands in a final group — the tree calculates the **average price** of all houses in that group and uses it as the prediction.

For example, if 4 houses are in a group with prices:
- $28,000,000
- $30,000,000
- $31,000,000
- $31,000,000

```
Average = (28 + 30 + 31 + 31) ÷ 4 = $30,000,000
```

---

## 📊 The Dataset

This project uses the **California Housing Dataset** which contains information about houses in California.

| Feature | What it means |
|---|---|
| `longitude` | How far left or right the house is on a map |
| `latitude` | How far up or down the house is on a map |
| `housing_median_age` | How old the houses in that area are |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | How many people live in that area |
| `households` | How many families live in that area |
| `median_income` | The average income of people in that area |
| `median_house_value` | The house price we want to predict ← TARGET |

### Quick Look at the Data

```python
df_dec.head()
```

---

## 🔧 Data Preparation

### Step 1 — Remove Missing Values

```python
# Drop missing values
df_dec = df_dec.dropna()
```

Any row with even one empty cell gets removed — the computer cannot learn from incomplete information.

### Step 2 — Separate Features and Target

```python
# Features (X) — the clues we give the model
X = df_dec[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income']]

# Target (y) — the answer we want to predict
y = df_dec['median_house_value']
```

| Variable | What it is |
|---|---|
| `X` | All the clues about each house |
| `y` | The house price we want to predict |

### Step 3 — Split Into Training and Testing Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

| Parameter | What it means |
|---|---|
| `test_size=0.2` | Use 20% of data for testing — 80% for training |
| `random_state=42` | Keep the split consistent every time |
| `X_train / y_train` | Data the model learns from (80%) |
| `X_test / y_test` | Data used to test how well it learned (20%) |

Think of it like this — the training data is the textbook the student studies, and the test data is the real exam with questions they have never seen before.

---

## 🌳 Building the Decision Tree Model

### Hyperparameters

Hyperparameters are settings YOU choose before training to control how the tree behaves. They do not come from the data.

```python
# Create and train the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(
    max_depth=10,           # Limit depth to prevent overfitting
    min_samples_split=20,   # Minimum samples required to split a node
    min_samples_leaf=10,    # Minimum samples required at a leaf node
    random_state=42
)

dt_regressor.fit(X_train, y_train)
```

| Hyperparameter | What it does |
|---|---|
| `max_depth=10` | Tree can ask at most 10 levels of questions — prevents memorizing |
| `min_samples_split=20` | Only split a group if it has at least 20 houses |
| `min_samples_leaf=10` | Every final answer must be based on at least 10 houses |
| `random_state=42` | Keep results the same every time |

### What is Overfitting?

Overfitting happens when the tree goes too deep and **memorizes** the training data instead of learning general patterns.

| | Tree Too Deep | Tree with max_depth=10 |
|---|---|---|
| What it does | Memorizes every detail | Learns general patterns |
| On training data | Almost perfect | Good but not perfect |
| On NEW data | Performs very badly | Performs well |

### Making Predictions

```python
# Predictions for training data
y_pred_train = dt_regressor.predict(X_train)

# Predictions for test data
y_pred_test = dt_regressor.predict(X_test)
```

### Viewing Results

```python
# Create a DataFrame for actual vs predicted values
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
display(results_df.head(10))
```

---

## 📈 Decision Tree Results

### Metrics

```python
# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("Training R² Score:", round(train_r2, 4))
print("Test R² Score:", round(test_r2, 4))
print("Training MAE:", round(train_mae, 2))
print("Test MAE:", round(test_mae, 2))
```

```
Training R² Score: 0.8054
Test R² Score:     0.7312
Training MAE:      34361.28
Test MAE:          40685.62
```

### Understanding the Metrics

**R² Score** — measures how well the model explains patterns in the data

| R² Score | What it means |
|---|---|
| 0.9 — 1.0 | Excellent 🎉 |
| 0.8 — 0.9 | Good ✅ |
| 0.7 — 0.8 | Okay ⚠️ |
| Below 0.7 | Poor ❌ |

**MAE (Mean Absolute Error)** — the average dollar amount the predictions are wrong by. Always convert to percentage to properly judge it:

```python
average_house_price = df_dec['median_house_value'].mean()
MAE_percentage_of_avg = (test_mae / average_house_price) * 100
print(round(MAE_percentage_of_avg, 2))
```

```
19.67
```

| MAE Percentage | What it means |
|---|---|
| 0% — 5% | Excellent 🎉 |
| 5% — 10% | Good ✅ |
| 10% — 20% | Okay ⚠️ |
| 20%+ | Poor ❌ |

**Overfitting Gap** — the difference between training and test performance

| | Score |
|---|---|
| Training R² | 80% |
| Test R² | 73% |
| Gap | 7% — noticeable overfitting ⚠️ |

### Visualizing Predictions

```python
# Scatter plot — Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Decision Tree Predictions vs Actual (Test Set)")
plt.grid(True)
plt.show()
```

> A perfect model would show all dots on a straight diagonal line from bottom-left to top-right. The closer the dots are to that diagonal — the better the model.

### Visualizing the Tree

```python
# Visualize first 3 levels of the tree
plt.figure(figsize=(20, 10))
features = X.columns
tree.plot_tree(dt_regressor,
               feature_names=features,
               max_depth=3,
               filled=True,
               rounded=True,
               fontsize=10)
plt.title('Decision Tree Visualization (First 3 Levels)')
plt.show()
```

Each box in the tree contains:

| Line in box | What it means |
|---|---|
| `median_income <= 5.035` | The question being asked |
| `squared_error = ...` | How mixed up the house prices are in this group |
| `samples = 16346` | How many houses are in this group |
| `value = 206644.4` | The predicted price if the tree stopped here |

---

## ⭐ Feature Importance — Decision Tree

```python
# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_regressor.feature_importances_
}).sort_values('importance', ascending=False)

display(feature_importance)
```

```
feature                importance
median_income          0.597315
longitude              0.171375
latitude               0.149376
housing_median_age     0.052824
population             0.011115
households             0.006720
total_bedrooms         0.006472
total_rooms            0.004802
```

```python
# Visualize feature importance
sorted_importance = feature_importance.sort_values(by='importance', ascending=True)

plt.figure(figsize=(10, 5))
plt.barh(sorted_importance['feature'], sorted_importance['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree Regressor')
plt.show()
```

### What This Tells Us

| Feature | Importance | Simple Meaning |
|---|---|---|
| `median_income` | ~60% | Rich area = expensive house ⭐ Most powerful |
| `longitude + latitude` | ~32% | Location matters a lot |
| `housing_median_age` | ~5% | Newer houses cost slightly more |
| Bottom 4 features | Less than 1% each | Barely useful to the tree |

---

## 🌲 Random Forest Regression

Random Forest builds **many Decision Trees** and combines their predictions to be more accurate and stable.

### How it Works

| Step | What happens |
|---|---|
| **1. Many Trees** | Builds 100 individual Decision Trees |
| **2. Bootstrapping** | Each tree gets its own random sample of the data |
| **3. Random Features** | Each tree only uses a random selection of features at each split |
| **4. Averaging** | Final prediction = average of all 100 trees' predictions |

### Why Random Forest is Better Than One Tree

Think of it like asking 100 different people for their opinion instead of just one person. Some people might be wrong — but the **average opinion of 100 people** is much more reliable than one person's opinion alone.

| | Single Decision Tree | Random Forest |
|---|---|---|
| Number of trees | 1 | 100 |
| Overfitting risk | High ❌ | Much lower ✅ |
| Accuracy | Okay | Much better ✅ |
| Consistency | Inconsistent | Much more consistent ✅ |

### What is Bootstrapping?

Bootstrapping means giving each tree its own **random sample** of the data — and allowing the same house to appear more than once.

```
Tree 1 learns from → House 1, 3, 3, 5, 6, 7, 7, 8, 9, 10
Tree 2 learns from → House 2, 2, 4, 5, 5, 6, 7, 8, 9, 10
Tree 3 learns from → House 1, 2, 3, 4, 6, 6, 8, 8, 9, 10
```

This makes each tree unique so their mistakes cancel out when averaged. ✅

---

## 🌲 Building the Random Forest Model

```python
# Create and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,      # Number of decision trees in the forest
    max_depth=10,          # Limit depth of each tree
    min_samples_split=20,  # Minimum samples required to split a node
    min_samples_leaf=10,   # Minimum samples required at a leaf node
    random_state=42,
    n_jobs=-1              # Use all available CPU cores
)

rf_regressor.fit(X_train, y_train)
```

| Hyperparameter | What it does |
|---|---|
| `n_estimators=100` | Build 100 Decision Trees in the forest |
| `max_depth=10` | Each tree can ask at most 10 levels of questions |
| `min_samples_split=20` | Only split a group with at least 20 houses |
| `min_samples_leaf=10` | Every final answer based on at least 10 houses |
| `random_state=42` | Keep results consistent every time |
| `n_jobs=-1` | Use ALL available CPU cores — builds trees faster |

### Making Predictions

```python
# Predictions for training data
y_pred_train_rf = rf_regressor.predict(X_train)

# Predictions for test data
y_pred_test_rf = rf_regressor.predict(X_test)

print("Random Forest Predictions (Training Set - first 10):")
print(y_pred_train_rf[:10])
print("\nRandom Forest Predictions (Test Set - first 10):")
print(y_pred_test_rf[:10])
```

---

## 📈 Random Forest Results

### Metrics

```python
# Evaluate Random Forest model
rf_train_r2 = r2_score(y_train, y_pred_train_rf)
rf_test_r2 = r2_score(y_test, y_pred_test_rf)
rf_train_mae = mean_absolute_error(y_train, y_pred_train_rf)
rf_test_mae = mean_absolute_error(y_test, y_pred_test_rf)

print("Training R² Score:", round(rf_train_r2, 4))
print("Test R² Score:", round(rf_test_r2, 4))
print("Training MAE:", round(rf_train_mae, 2))
print("Test MAE:", round(rf_test_mae, 2))
```

```
Training R² Score: 0.8308
Test R² Score:     0.7791
Training MAE:      31872.57
Test MAE:          36640.07
```

### MAE Percentage

```python
average_house_price = df_dec['median_house_value'].mean()
MAE_percentage_of_avg = (rf_test_mae / average_house_price) * 100
print(round(MAE_percentage_of_avg, 2))
```

```
17.71
```

### Visualizing Predictions

```python
# Scatter plot — Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_test_rf, alpha=0.3)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Random Forest Predictions vs Actual (Test Set)")
plt.grid(True)
plt.show()
```

### Visualizing One Tree From the Forest

```python
plt.figure(figsize=(20, 10))
features = X.columns
tree.plot_tree(rf_regressor.estimators_[0],
               feature_names=features,
               max_depth=3,
               filled=True,
               rounded=True,
               fontsize=10)
plt.title('Random Forest - Visualization of the First Tree (First 3 Levels)')
plt.show()
```

> `estimators_[0]` means "show me the first tree from the forest." We can only look at one tree at a time since showing all 100 is impossible.

---

## ⭐ Feature Importance — Random Forest

```python
# Feature importance for Random Forest
rf_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_regressor.feature_importances_
}).sort_values('importance', ascending=False)

display(rf_feature_importance)
```

```
feature                importance
median_income          0.601485
longitude              0.161497
latitude               0.146397
housing_median_age     0.054576
population             0.012380
total_bedrooms         0.011139
total_rooms            0.006672
households             0.005855
```

```python
# Visualize feature importance
sorted_importance_rf = rf_feature_importance.sort_values(by='importance', ascending=True)

plt.figure(figsize=(10, 5))
plt.barh(sorted_importance_rf['feature'], sorted_importance_rf['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Regressor')
plt.show()
```

---

## 🏆 Model Comparison

### R² Score Comparison

| | Decision Tree | Random Forest | Winner |
|---|---|---|---|
| **Training R²** | 0.8054 (80%) | 0.8308 (83%) | 🌲 Random Forest |
| **Test R²** | 0.7312 (73%) | 0.7791 (78%) | 🌲 Random Forest |

### MAE Comparison

| | Decision Tree | Random Forest | Winner |
|---|---|---|---|
| **Training MAE** | $34,361 | $31,872 | 🌲 Random Forest |
| **Test MAE** | $40,685 | $36,640 | 🌲 Random Forest |

### MAE Percentage Comparison

| Model | MAE Percentage | Rating |
|---|---|---|
| **Decision Tree** | 19.67% | ⚠️ Almost failing |
| **Random Forest** | 17.71% | ⚠️ Better but room to improve |

### Overfitting Gap Comparison

| | Training R² | Test R² | Gap |
|---|---|---|---|
| **Decision Tree** | 80% | 73% | 7% gap ⚠️ |
| **Random Forest** | 83% | 78% | 5% gap ✅ |

> Random Forest wins on every single metric — higher R² scores, lower MAE errors, and less overfitting.

### Feature Importance Comparison

Both models agreed on the same ranking 👇

| Feature | Decision Tree | Random Forest |
|---|---|---|
| `median_income` | 59.7% ⭐ | 60.1% ⭐ |
| `longitude` | 17.1% | 16.1% |
| `latitude` | 14.9% | 14.6% |
| `housing_median_age` | 5.3% | 5.5% |
| Bottom 4 features | Less than 1% each | Less than 1% each |

> When one tree and 100 trees independently reach the same conclusion — you can be very confident that conclusion is correct.

### The Big Story the Data Tells Us

```
60% of house price prediction → depends on how wealthy the area is
32% of house price prediction → depends on where the house is located
 5% of house price prediction → depends on how old the houses are
 3% of house price prediction → everything else combined
```

---

## 📖 Key Concepts Glossary

| Concept | Simple Meaning |
|---|---|
| **Machine Learning** | Teaching a computer to learn from examples |
| **Regression** | Predicting a number |
| **Feature** | A piece of information about each house |
| **Target** | The answer we want to predict |
| **Split** | Dividing a group into two smaller groups based on a yes/no question |
| **Root Node** | The very first question — where the tree starts |
| **Internal Node** | Questions in the middle of the tree |
| **Leaf Node** | The final answer at the bottom — a predicted price |
| **Branch** | The YES or NO path connecting nodes |
| **Overfitting** | When a model memorizes instead of learns |
| **Hyperparameter** | Settings YOU choose before training to control the model |
| **max_depth** | How many levels of questions the tree can ask |
| **min_samples_split** | Minimum houses needed before splitting a group |
| **min_samples_leaf** | Minimum houses needed at a leaf node |
| **R² Score** | How well the model explains patterns — closer to 1.0 is better |
| **MAE** | Average dollar amount predictions are wrong by — lower is better |
| **MAE Percentage** | MAE as a percentage of average price — stay below 20% |
| **Bootstrapping** | Giving each tree a different random sample of data |
| **n_estimators** | How many trees to build in the Random Forest |
| **n_jobs=-1** | Use all available CPU cores for faster training |
| **Feature Importance** | Which features were most useful for making predictions |

---

## 🛠️ Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import tree
```

---

## 🚀 How to Run

1. Clone the repository
2. Install required libraries
3. Open the notebook in Google Colab or Jupyter Notebook
4. Run all cells from top to bottom

---

*Built with ❤️ while learning Data Science*
