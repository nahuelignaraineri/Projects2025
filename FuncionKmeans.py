def aplicar_clustering_y_etiquetar(df, n_clusters=4, n_init=1000, algoritmo="elkan", random_state=15):
    """
    Aplica el algoritmo K-means a un dataframe, ordena los clusters por valores de impedancia
    y etiqueta clusters específicos
    
    Parámetros:
    -----------
    df_sc : pandas.DataFrame
        DataFrame con los datos a agrupar
    n_clusters : int, opcional (por defecto=4)
        Número de clusters a crear
    n_init : int, opcional (por defecto=1000)
        Número de inicializaciones del algoritmo
    algoritmo : str, opcional (por defecto="elkan")
        Algoritmo a utilizar para K-means ("elkan" o "lloyd")
    random_state : int, opcional (por defecto=15)
        Semilla para reproducibilidad
        
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con la columna de clusters añadida y etiquetada
    dict
        Diccionario con los centroides de los clusters
    """
    
    import pandas as pd
    import seaborn as sns
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Copia del dataframe para no modificar el original
    df_resultado = df.copy()
    
    # Aplicar el algoritmo K-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, algorithm=algoritmo, random_state=random_state).fit(
        df_resultado
    )
    centroids = kmeans.cluster_centers_
    
    # Añadir la columna de clusters al dataframe
    df_resultado["Cluster"] = kmeans.labels_
    
    # Ordenar los clusters basados en los valores de una columna específica
    cluster_order = df_resultado.groupby("Cluster")["Vp/Vs"].mean().sort_values(ascending=False).index
    cluster_mapping = {old: new for new, old in enumerate(cluster_order)}
    df_resultado["Cluster"] = df_resultado["Cluster"].map(cluster_mapping)
    
    # Etiquetado si es necesario
    df_resultado.loc[df_resultado["Cluster"].isin([0, 1, 2]), "Cluster"] = 0
    df_resultado.loc[df_resultado["Cluster"] == 3, "Cluster"] = 1
    
    return df_resultado, centroids

# Ejemplo de uso:
# df_resultado, centroids = aplicar_clustering_y_etiquetar(DF_SC)
# 
# # Graficar el pairplot con seaborn
# sns.pairplot(df_resultado, hue="Cluster", diag_kind="kde", palette="tab10")
# plt.show()