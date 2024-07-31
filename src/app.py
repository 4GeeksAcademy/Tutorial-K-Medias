from utils import db_connect
engine = db_connect()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar el conjunto de datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
data = pd.read_csv(url)

# Seleccionar solo las columnas de interés
data = data[['Latitude', 'Longitude', 'MedInc']]

# Dividir el conjunto de datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Visualizar las primeras filas del conjunto de entrenamiento y prueba
print(train_data.head())
print(test_data.head())

# Construir el modelo K-Means
kmeans_model = KMeans(n_clusters=6, random_state=42)
train_data['Cluster'] = kmeans_model.fit_predict(train_data[['Latitude', 'Longitude', 'MedInc']])
test_data['Cluster'] = kmeans_model.predict(test_data[['Latitude', 'Longitude', 'MedInc']])

# Guardar el modelo K-Means
joblib.dump(kmeans_model, 'kmeans_model.pkl')

# Visualizar los clusters
plt.figure(figsize=(10, 6))
plt.scatter(train_data['Longitude'], train_data['Latitude'], c=train_data['Cluster'], cmap='viridis', label='Train Data', alpha=0.5)
plt.scatter(test_data['Longitude'], test_data['Latitude'], c=test_data['Cluster'], cmap='viridis', label='Test Data', edgecolor='k', alpha=0.5)
plt.colorbar(label='Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clusters de casas (Train y Test)')
plt.legend()
plt.show()

# Entrenar un modelo de clasificación supervisada
from sklearn.ensemble import RandomForestClassifier

# Preparar los datos para el modelo supervisado
X_train = train_data[['Latitude', 'Longitude', 'MedInc']]
y_train = train_data['Cluster']
X_test = test_data[['Latitude', 'Longitude', 'MedInc']]
y_test = test_data['Cluster']

# Entrenar el modelo
supervised_model = RandomForestClassifier(random_state=42)
supervised_model.fit(X_train, y_train)

# Guardar el modelo supervisado
joblib.dump(supervised_model, 'supervised_model.pkl')

# Predicciones y evaluación del modelo supervisado
y_pred = supervised_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualizar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión del Modelo Supervisado')
plt.show()

# Índice de silueta para determinar el número óptimo de clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data[['Latitude', 'Longitude', 'MedInc']])
    silhouette_scores.append(silhouette_score(data[['Latitude', 'Longitude', 'MedInc']], cluster_labels))

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Índice de Silueta')
plt.title('Optimización del Número de Clústeres')
plt.show()
