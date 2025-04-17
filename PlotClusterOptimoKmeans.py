from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from typing import Optional, Union


def PlotClusterOptimo(
    df: pd.DataFrame, show: Optional[bool] = True, n_cluster_max: Optional[int] = 10, path: Optional[str] = None
):
    """Función que grafica dos subplots: uno con la suma de la distancia entre
     los centroides y los elementos del cluster; otro con el coeficiente de
    silhouette. Se usan para elegir el número optimo de clusteres con el que se
    deben agrupar los datos.

    Args:
        df (pd.DataFrame): Dataframe con datos a clusterizar, solo admite datos
        numéricos, el index puede ser string.
        show (Optional[bool], optional): True o False para mostrar el gráfico. Defaults to True.
        n_cluster_max (int, optional): N° de cluster máximo a investigar. Defaults to 10.
        path (_type_, optional): Ruta donde se quiere guardar el gráfico. Defaults to None.
    """
    sse = {}
    for k in range(1, n_cluster_max):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df)
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[k] = kmeans.inertia_

    silhouette = {}
    for n_cluster in range(2, n_cluster_max):
        kmeans = KMeans(n_clusters=n_cluster, random_state=16).fit(df)
        label = kmeans.labels_
        silhouette[n_cluster] = silhouette_score(df, label, metric="euclidean")

    fig = plt.figure(figsize=(16, 6))
    fig.add_subplot(1, 2, 1)
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster", fontsize=14)
    plt.ylabel("SSE", fontsize=14)
    plt.title("Minimizar distancia (SSE)", fontsize=20)
    # plt.grid()

    fig.add_subplot(1, 2, 2)
    plt.plot(list(silhouette.keys()), list(silhouette.values()))
    plt.xlabel("Number of cluster", fontsize=14)
    plt.ylabel("Silhouette Coefficient", fontsize=14)
    plt.title("Silhouette Coefficient", fontsize=20)
    # plt.grid()

    if path is not None:
        try:
            fig.savefig(path + "\\" + f"Plot_N_Clusteres.png")
        except:
            print(
                "No se puede acceder a la ruta especificada para guardar \
los gráficos."
            )

    if show:
        plt.show()
    return