import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
triggers = pd.read_csv("triggers.csv")
actions = pd.read_csv("actions.csv")

# Преобразование форматов и типов
triggers['date'] = pd.to_datetime(triggers['date'], errors='coerce')
actions['date'] = pd.to_datetime(actions['date'], errors='coerce')
actions['guid'] = actions['guid'].astype(str)
triggers['guid'] = triggers['guid'].astype(str)

# Объединение данных с учетом всех триггеров до действия
merged = pd.merge(
    actions, 
    triggers, 
    on='guid', 
    suffixes=('_action', '_trigger'),
    how='left'
)
merged = merged[merged['date_trigger'] <= merged['date_action']]

# Агрегация триггеров для каждого действия
aggregated = merged.groupby(['guid', 'date_action'], group_keys=False).agg(
    trigger_count=('date_trigger', 'count'),  # Общее количество триггеров
    last_trigger=('date_trigger', 'max'),     # Дата последнего триггера
    first_trigger=('date_trigger', 'min')     # Дата первого триггера
).reset_index()

# Вычисление производных признаков
aggregated['last_trigger_days'] = (aggregated['date_action'] - aggregated['last_trigger']).dt.days
aggregated['trigger_freq'] = (
    (aggregated['last_trigger'] - aggregated['first_trigger']).dt.days 
    / aggregated['trigger_count']
).replace(np.inf, 0).fillna(0)

# Объединяем агрегированные данные с исходными действиями
filtered_data = pd.merge(
    actions, 
    aggregated,
    left_on=['guid', 'date'], 
    right_on=['guid', 'date_action'], 
    how='left'
)

# Заполняем пропуски для пользователей без триггеров
filtered_data['trigger_count'] = filtered_data['trigger_count'].fillna(0)
filtered_data['last_trigger_days'] = filtered_data['last_trigger_days'].fillna(-1)  # -1 означает отсутствие триггеров
filtered_data['trigger_freq'] = filtered_data['trigger_freq'].fillna(0)

# Создаем столбец days_since_last, если он отсутствует
if 'days_since_last' not in filtered_data.columns:
    filtered_data['days_since_last'] = filtered_data.groupby('guid')['date'].diff().dt.days.fillna(0)

# Удаление повторных показов (оставляем только последнее действие)
filtered_data = filtered_data.sort_values(by=['guid', 'date'])
filtered_data = filtered_data.drop_duplicates(subset='guid', keep='last')

# Убедимся, что данные не пустые после фильтрации
if filtered_data.empty:
    raise ValueError("Нет данных для анализа после фильтрации.")

# Проверим, что столбец 'result' существует и содержит более одного класса
if 'result' not in filtered_data.columns:
    raise ValueError("Столбец 'result' отсутствует в данных.")
if len(filtered_data['result'].unique()) < 2:
    raise ValueError("Целевая переменная должна содержать хотя бы два уникальных класса.")

# Добавим дополнительные признаки для обучения
filtered_data['action_count'] = filtered_data.groupby('guid')['guid'].transform('count')  # Количество действий пользователя
filtered_data['days_since_start'] = (filtered_data['date'] - filtered_data['date'].min()).dt.days  # Дни с начала анализа
filtered_data['action_per_trigger'] = filtered_data['action_count'] / (filtered_data['trigger_count'] + 1e-6)  # Защита от деления на ноль
filtered_data['is_triggered'] = (filtered_data['trigger_count'] > 0).astype(int)

# Определим признаки для обучения
features = [
    'days_since_last', 
    'action_count', 
    'days_since_start',
    'trigger_count', 
    'last_trigger_days',
    'trigger_freq',
    'action_per_trigger',
    'is_triggered'
]
X = filtered_data[features]
y = filtered_data['result']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Используем SMOTE для балансировки классов
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Обучение модели RandomForest
model = RandomForestClassifier(
    class_weight={0: 1, 1: 3},  # Увеличиваем вес класса 1
    n_estimators=100,
    random_state=42
)
model.fit(X_train_smote, y_train_smote)

# Оптимизация порога классификации по ROC-кривой
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
y_pred = (model.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)

# Оценка модели
print("Отчет о классификации:")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC-AUC метрика
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC: {roc_auc:.2f}")

# Построение ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Матрица ошибок
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Финансовые метрики
cost_for_one_show = 1
get_money_for_user_click = 5

# Предсказания вероятностей на полном наборе данных
filtered_data['predicted_probability'] = model.predict_proba(X)[:, 1]
filtered_data['predicted_result'] = (filtered_data['predicted_probability'] >= optimal_threshold).astype(int)

# Расчет финансовых метрик
filtered_data['cost'] = cost_for_one_show
filtered_data['revenue'] = filtered_data['predicted_result'] * get_money_for_user_click
filtered_data['balance'] = filtered_data['revenue'] - filtered_data['cost']

# Фильтрация для положительного баланса
positive_balance_data = filtered_data[filtered_data['balance'] > 0]

if positive_balance_data.empty:
    print("Нет записей с положительным балансом после расчёта.")
else:
    # Сортировка данных по убыванию дохода
    positive_balance_data = positive_balance_data.sort_values(by='revenue', ascending=False)

    # Общие финансовые результаты
    total_cost = positive_balance_data['cost'].sum()
    total_revenue = positive_balance_data['revenue'].sum()
    total_balance = total_revenue - total_cost

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"Для положительного баланса получаем следующие результаты:")
    print(f"Общие расходы: {total_cost:.2f} доллар(-ов).")
    print(f"Общий доход: {total_revenue:.2f} доллар(-ов).")
    print(f"Итоговый баланс: {total_balance:.2f} доллар(-ов).")


importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Важность признаков")
plt.show()
