#Imports Necesarios

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs


#Creacion de Datos Sintéticos
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

X.shape

#Ver los datos sintéticos generados
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
plt.title("Dataset sintético de mostra")
plt.show()
import math

def DistEuclidiana(x1, y1, x2, y2):
    '''
    x1 = X[0]
    x2 = X[1]
    y1 = X[0]
    y2 = X[1]
    
    '''
    distancia = 100000000000
    if math.sqrt((x2 - x1)**2 + (y2 - y1)**2) < distancia:
        distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    elif math.sqrt((x2 - x1)**2 + (y2 - y1)**2) == distancia:
        # Ver cuantos puntos tiene cada centroide y asignarlo y elegir ese

