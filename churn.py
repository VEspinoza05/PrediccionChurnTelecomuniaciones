import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

RND = 42
np.random.seed(RND)

# -------------------------
# 1) Generar dataset
# -------------------------
n = 10000
df = pd.DataFrame({
    "SeniorCitizen": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    "Tenure": np.random.randint(0, 72, size=n),
    "MonthlyCharges": np.round(np.random.uniform(20, 140, size=n), 2),

    "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.20]),
    "TechSupport": np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70]),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]),
    "PaymentMethod": np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n
    ),

    "NumComplaints": np.random.poisson(0.25, size=n),
    "AvgDailyUsageHours": np.round(np.random.exponential(scale=1.5, size=n), 2),
})

df["TotalCharges"] = np.round((df["Tenure"] * df["MonthlyCharges"]) + np.random.normal(0, 50, size=n), 2)
df["TotalCharges"] = df["TotalCharges"].clip(lower=0)

# Exportacion a CSV para trazabilidad
df.to_csv("churn.csv", index=False)

# Modelo probabilístico realista
coef = {
    "intercept": -1.2,
    "senior": 0.6,
    "tenure": -0.03,
    "monthly": 0.01,
    "fiber": 0.9,
    "no_internet": -0.4,
    "tech_support_yes": -0.8,
    "month_to_month": 1.1,
    "one_year": -0.3,
    "num_complaints": 0.7,
    "usage": 0.05
}

score = np.full(n, coef["intercept"])
score += coef["senior"] * df["SeniorCitizen"]
score += coef["tenure"] * df["Tenure"]
score += coef["monthly"] * df["MonthlyCharges"]
score += coef["num_complaints"] * df["NumComplaints"]
score += coef["usage"] * df["AvgDailyUsageHours"]

score += np.where(df["InternetService"] == "Fiber optic", coef["fiber"], 0)
score += np.where(df["InternetService"] == "No", coef["no_internet"], 0)
score += np.where(df["TechSupport"] == "Yes", coef["tech_support_yes"], 0)
score += np.where(df["Contract"] == "Month-to-month", coef["month_to_month"],
                  np.where(df["Contract"] == "One year", coef["one_year"], 0))

score += np.random.normal(0, 0.8, size=n)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


prob = sigmoid(score)
df["Churn"] = (np.random.rand(n) < prob).astype(int)

# -------------------------
# 2) Split
# -------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RND, stratify=y
)

# -------------------------
# 3) Preprocesamiento
# -------------------------
cat_cols = ["InternetService", "TechSupport", "Contract", "PaymentMethod"]
num_cols = ["SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges",
            "NumComplaints", "AvgDailyUsageHours"]

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# -------------------------
# 4) Modelo
# -------------------------
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=RND
)

pipeline = Pipeline([
    ("preproc", preproc),
    ("smote", SMOTE(random_state=RND)),
    ("clf", xgb)
])

pipeline.fit(X_train, y_train)

y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# -------------------------
# 5) Métricas
# -------------------------
roc_auc = roc_auc_score(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

print(f"ROC AUC = {roc_auc:.4f}")
print(f"PR-AUC = {ap:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# Precision-Recall
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 6))
plt.plot(rec, prec, label=f"AP={ap:.3f}")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# 6) Gráfico de barras de Churn
# -------------------------
plt.figure(figsize=(6, 4))
df["Churn"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribución de Churn")
plt.xlabel("Churn (0 = No, 1 = Sí)")
plt.ylabel("Cantidad")
plt.grid(axis="y")
plt.show()

# -------------------------
# 7) Boxplot MonthlyCharges vs Churn
# -------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Distribución de MonthlyCharges según Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.grid(axis="y")
plt.show()