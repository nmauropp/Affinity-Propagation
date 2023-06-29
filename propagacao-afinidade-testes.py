# #############################################################################
# Imports - Affinity Propagation
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

# #############################################################################
# Exemplo de clusterização com Propagação de Afinidade

# #############################################################################
# Define 4 pontos distantes para os centros (pontos preferenciais)

centros = [[4, 1], [-5, 4], [-1, -1], [-2, -1]]
# Recebe a matriz e a validação de rótulos
matriz, labels_true = make_blobs(n_samples=300, centers=centros, cluster_std=0.5,
                            random_state=0)

# #############################################################################
# Aplica a Affinity Propagation

af = AffinityPropagation().fit(matriz)
# Recebe os indices dos clusters após o algoritmo
cluster_centers_indices = af.cluster_centers_indices_
# Recebe os rótulos
labels = af.labels_
# Recebe a quantidade de clusters
n_clusters_ = len(cluster_centers_indices)


# #############################################################################
# Imports - Plot
import matplotlib.pyplot as plt
from itertools import cycle

# #############################################################################
# Definições do Plot

plt.close('all')
plt.figure(1)
plt.clf()
# Tamanho do quadro
ax = plt.axes([0, 0, 1.5, 1.5])
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = matriz[cluster_centers_indices[k]]
    plt.plot(matriz[class_members, 0], matriz[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in matriz[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
plt.title('Quando a Entrada possui 4 pontos preferenciais relativamente distantes:\n\nNúmero de Clusters = %d\n (mesmo número de centros)' % n_clusters_)
plt.show()

