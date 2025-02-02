import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Loading data
triggers = pd.read_csv("triggers.csv")
actions = pd.read_csv("actions.csv")

# Convert formats and types
triggers['date'] = pd.to_datetime(triggers['date'], errors='coerce')
actions['date'] = pd.to_datetime(actions['date'], errors='coerce')
actions['guid'] = actions['guid'].astype(str)
triggers['guid'] = triggers['guid'].astype(str)

# Counting the number of triggers and the average interval between triggers for each user
triggers = triggers.sort_values(by=['guid', 'date'])
triggers['prev_trigger_date'] = triggers.groupby('guid')['date'].shift(1)
triggers['days_between_triggers'] = (triggers['date'] - triggers['prev_trigger_date']).dt.days
triggers['days_between_triggers'] = triggers['days_between_triggers'].fillna(0)

# Aggregation by user
trigger_stats = triggers.groupby('guid').agg(
    trigger_count=('guid', 'count'),  
    avg_days_between_triggers=('days_between_triggers', 'mean')
).reset_index()

# Data merge
data = pd.merge(actions, triggers, on='guid', how='left')
data = pd.merge(data, trigger_stats, on='guid', how='left')
data['date'] = data['date_x'].fillna(data['date_y'])

# Deleting repeat shows (every 2 weeks)
data = data.sort_values(by=['guid', 'date'])
data['previous_date'] = data.groupby('guid')['date'].shift(1)
data['days_since_last'] = (data['date'] - data['previous_date']).dt.days
filtered_data = data[(data['days_since_last'].isna()) | (data['days_since_last'] > 14)].copy()

# Check if data is available after filtering
if filtered_data.empty:
    raise ValueError("No data to analyze after filtration.")

# Filling in missing values
filtered_data['days_since_last'] = filtered_data['days_since_last'].fillna(0)
filtered_data['trigger_count'] = filtered_data['trigger_count'].fillna(0)
filtered_data['avg_days_between_triggers'] = filtered_data['avg_days_between_triggers'].fillna(filtered_data['days_since_last'])

# Checking the target variable
if 'result' not in filtered_data.columns:
    raise ValueError("The 'result' column is missing from the data.")
if len(filtered_data['result'].unique()) < 2:
    raise ValueError("The target variable must contain at least two unique classes.")

# Additional features
filtered_data['action_count'] = filtered_data.groupby('guid')['guid'].transform('count')  
filtered_data['days_since_start'] = (filtered_data['date'] - filtered_data['date'].min()).dt.days  

# Defining features for the model
features = ['days_since_last', 'action_count', 'days_since_start', 'trigger_count', 'avg_days_between_triggers']
X = filtered_data[features]
y = filtered_data['result']

# Separating the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model training
model = LogisticRegression(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Estimating the model
print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC-AUC metric
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {roc_auc:.2f}")

# ROC curve construction
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Error matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Financial metrics
cost_for_one_show = 1
get_money_for_user_click = 5

# Probability predictions on the full data set
filtered_data['predicted_probability'] = model.predict_proba(X)[:, 1]
filtered_data['predicted_result'] = (filtered_data['predicted_probability'] >= 0.5).astype(int)

# Calculating financial metrics
filtered_data['cost'] = cost_for_one_show
filtered_data['revenue'] = filtered_data['predicted_result'] * get_money_for_user_click
filtered_data['balance'] = filtered_data['revenue'] - filtered_data['cost']

# Filtering positive balance
positive_balance_data = filtered_data[filtered_data['balance'] > 0]

if positive_balance_data.empty:
    print("No records with positive balance after settlement.")
else:
    # Sorting by descending income
    positive_balance_data = positive_balance_data.sort_values(by='revenue', ascending=False)

    # Final financial results
    total_cost = positive_balance_data['cost'].sum()
    total_revenue = positive_balance_data['revenue'].sum()
    total_balance = total_revenue - total_cost

    print("Error matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"For positive balance we obtain the following results:")
    print(f"Total costs: {total_cost:.2f}$.")
    print(f"Total revenue: {total_revenue:.2f}$.")
    print(f"Final balance: {total_balance:.2f}$.")

