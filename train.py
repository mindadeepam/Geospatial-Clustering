import pandas as pd
from sklearn.cluster import KMeans
import pickle

def main(coords, n_clusters, return_labels=True, save_model=True, model_path=r"models/kmeans.pkl"):

    kmeansModel = KMeans(n_clusters=n_clusters)
    kmeansModel.fit(coords)

    if(save_model):
        with open(model_path, "wb") as output_file:
            pickle.dump(kmeansModel, output_file)

    
    labels = kmeansModel.predict(coords)
    return labels, kmeansModel.cluster_centers_


    




