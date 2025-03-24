import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import make_blobs
import math

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()



plt.scatter(X[:, 0], X[:, 1], s=10)
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
plt.title("Dataset sintético de mostra")
plt.show()

def Kmeans(RandomPoints=False, iteracion_max = 50,k=4):
    centroides = []
    if RandomPoints:
        for _ in range(k):
            centroide = [random.uniform(x_min, x_max), random.uniform(y_min,y_max)]
            centroides.append(centroide)
    else:
        for _ in range(k):
            centroide= [random.choice(X)]
            centroides.append(centroide)

    for _ in range (iteracion_max):
        num_centroides = [0] * k
        clusters = [[] for _ in range(k)]

        for point in X:
            indice = DistEuclidiana(point, centroides, num_centroides)
            clusters[indice].append(point)

        nuevos_centroides = []
        for cluster in clusters:
            if cluster:
                nuevo_centroide = [sum(p[0] for p in cluster) / len(cluster), 
                                   sum(p[1] for p in cluster) / len(cluster)]
            else:
                # Si un cluster queda vacío, asignamos un punto aleatorio
                nuevo_centroide = random.choice(X)
            nuevos_centroides.append(nuevo_centroide)
            
        if nuevos_centroides == centroides:
            break
        
        centroides = nuevos_centroides  

        Graf(k, clusters, centroides)

def DistEuclidiana(a, b, num_centroides):
    x1 = a[0]
    x2 = a[1]
    distancia_min = math.inf
    mejor_centroide = -1
    for i in range(len(b)):
        y1 = b[i][0]
        y2 = b[i][1]
    
        distancia_actual = math.sqrt((y1 - x1)**2 + (y2 - x2)**2)
        if distancia_actual < distancia_min:
            distancia_min = distancia_actual
            mejor_centroide = i

        elif distancia_actual == distancia_min:
            if num_centroides[i] < num_centroides[mejor_centroide]:
                mejor_centroide = i
    num_centroides[mejor_centroide] += 1
    return mejor_centroide

def Graf(k,clusters,centroides):
    colores = ['r', 'g', 'b', 'y']
    for i in range(k):
        cluster = clusters[i]
        plt.scatter([p[0] for p in cluster], [p[1] for p in cluster], s=10, color=colores[i])
        plt.scatter(centroides[i][0], centroides[i][1], marker='x', color='k', s=100)
    
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    plt.title("K-Means Clustering")
    plt.show()


Kmeans(False, 50,4)
