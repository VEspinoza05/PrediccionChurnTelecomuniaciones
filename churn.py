import pandas as pd # Manejo de estructuras de datos tipo DataFrame
import numpy as np # Operaciones numéricas y generación de datos aleatorios
from sklearn.model_selection import train_test_split# División de datos en entrenamiento y prueba
from sklearn.ensemble import RandomForestClassifier # Clasificador basado en bosques
from sklearn.metrics import classification_report# Métricas de evaluación del modelo
import seaborn as sns # Visualización estadística
import matplotlib.pyplot as plt # Gráficos en general

#crear datos con una semilla
np.random.seed(42) # Fijar semilla para reproducibilidad
# Número de registros # Número de registros simulados
n = 5000
# Generar columnas simuladas
# Generar IDs únicos para clientes
customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n + 1)]
# Simular duración del contrato en meses (1 a 60)
tenure = np.random.randint(1, 61, size=n)  # entre 1 y 60 meses
# Simular cargos mensuales entre $30 y $120
monthly_charges = np.round(np.random.uniform(30, 120, size=n), 2)
# Calcular cargos totales como producto de tenure y cargos mensuales
total_charges = np.round(monthly_charges * tenure, 2)
# Simular tipo de contrato con probabilidades específicas
contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.6, 0.2, 0.2])
# Simular método de pago
payment_methods = np.random.choice([
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
], size=n)
# Simular variable objetivo: churn (abandono), con 30% de probabilidad
churn = np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7])  # 30% churn rate
# Crear DataFrame con todas las variables simuladas
# Crear DataFrame
df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contracts,
    "PaymentMethod": payment_methods,
    "Churn": churn
})
# Exportar a archivo CSV para trazabilidad
# Exportar a CSV
df.to_csv("churn.csv", index=False)

# Mostrar los primeros 5 registros
print(df.head())

# Cargar datos
df = pd.read_csv('churn.csv')

# # Limpieza básica
df = df.dropna(subset=["CustomerID","Tenure","MonthlyCharges","TotalCharges","Contract","PaymentMethod","Churn"])

# # Estadísticas básicas
print(df.describe())

# # Distribución de churn
print(df['Churn'].value_counts(normalize=True))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.show()

df_encoded = pd.get_dummies(df, drop_first=True)
# Separar variables predictoras (X) y objetivo (y)
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']
from imblearn.over_sampling import SMOTE # Técnica de sobremuestreo sintético
# Aplicar SMOTE para balancear clases
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
from sklearn.model_selection import train_test_split
# Separar datos en entrenamiento y prueba (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
from xgboost import XGBClassifier  # Modelo de clasificación basado en árboles optimizados
# Instanciar y entrenar el modelo
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)
# Matriz de confusión
print(confusion_matrix(y_test, y_pred))
# Métricas de clasificación (precisión, recall, F1)
print(classification_report(y_test, y_pred))

# Calcular AUC (Área bajo la curva ROC)
y_proba = model.predict_proba(X_test)[:,1]
print("AUC:", roc_auc_score(y_test, y_proba))
importances = model.feature_importances_
features = X.columns
# Crear DataFrame para visualizar importancia
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
# Mostrar las 10 variables más relevantes
print(feature_importance_df.sort_values(by='Importance', ascending=False).head(10))
