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