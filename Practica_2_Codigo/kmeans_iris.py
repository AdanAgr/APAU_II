
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import warnings

# Ignorar advertencias
warnings.filterwarnings("ignore")

# Cargar el dataset de Iris
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('iris.data', header=None, names=columns)

# Eliminar la columna de especies para el clustering
X = df.drop('species', axis=1)

# Escalar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar el número óptimo de clusters usando el método del codo
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('SSE')
plt.title('Método del codo')
plt.show()

# Usar KMeans con el número óptimo de clusters
optimal_k = KneeLocator(range(1, 11), sse, curve='convex', direction='decreasing').knee
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# Añadir los labels al dataframe original
df['cluster'] = kmeans.labels_

# Visualización de los clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='cluster', palette='tab10')
plt.title('Clusters visualizados (sepal_length vs sepal_width)')
plt.show()

# Métrica de evaluación: Silhouette Score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score para {optimal_k} clusters: {silhouette_avg:.2f}")
