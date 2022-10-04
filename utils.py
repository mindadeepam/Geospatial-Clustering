import pickle
import geopy.distance
import numpy as np
import pandas as pd
import train

def preprocess(df, train=False, predict=False):
    if train==True:
        ## remove null/0 values for lat long
        df = df.drop(df[df.longitude==0.0].index)
        df = df.loc[(df.longitude<=98) & (df.longitude>=60)]
        df = df.loc[(df.latitude<=38) & (df.latitude>=8)]
      
        poi_cols = [ "latitude","longitude", "state", "cityData", "chapterData"]
        df = df.groupby(['student_id','question_id']).first()[poi_cols].reset_index()  

    if predict==True:
        df = df[["student_id", "latitude", "longitude"]]
        # pass

    return df


def top_chapters(self):
    n = 5
    return self.value_counts().head(n).to_dict()


def top_city(self):
    self = self.mode()
    self = self.apply(lambda x: x[0] if type(x)!=str else x )
    return self


def get_radius_of_clusters(kmeans_cluster_centres, df2):
  radius_of_cluster = []
  for i, k in enumerate(range(len(kmeans_cluster_centres))):
    uniq_users = df2[df2.kmeans_cluster == k].groupby('student_id').first().loc[:, ['longitude', "latitude" ]]
    center = kmeans_cluster_centres[k]
    distances = []
    for lon, lat in uniq_users.values:
      distances.append(geopy.distance.geodesic((center[1],center[0]), (lat, lon)).km)
    radius_of_cluster.append(np.mean(sorted(distances, reverse=True)[:5]))
  return np.round_(radius_of_cluster, 2)


def getClusterdf(df, cluster_centers):
    df.set_index("kmeans_cluster", drop=True).sort_index()
    cluster_based_df = df.groupby("kmeans_cluster").agg({
    'cityData': top_city, 
    "chapterData": top_chapters, 
    "student_id": pd.Series.nunique
    })

    cluster_based_df.cityData = cluster_based_df.cityData.apply(lambda x: x[0] if type(x)!=str else x)
    cluster_based_df.columns = ["city", "top_chapters", "cluster_population"]

    ## radius values of clusters
    radius_of_cluster = get_radius_of_clusters(cluster_centers, df)
    radius_in_kms = pd.Series(radius_of_cluster, name="radius_in_kms")
    cluster_based_df = pd.concat((cluster_based_df, radius_in_kms), axis=1)
    cluster_based_df.index.name = 'cluster'

    cluster_based_df["center_lon"] = cluster_centers[:,0]
    cluster_based_df["center_lat"] = cluster_centers[:,1]
    return cluster_based_df



def findClosestCentre(x, centres, indices):
    distances = []
    lon, lat = x[0], x[1]
    for center_lon, center_lat in centres:
        distances.append(geopy.distance.geodesic((center_lat, center_lon), (lat, lon)).km)
    i = np.argmin(distances)
    return indices[i]

def getClusters(df, cluster_df):
    filtered_indices = cluster_df.index
    filtered_centers = cluster_df[["center_lon", "center_lat"]].values
    labels = df.apply(findClosestCentre, args=(filtered_centers, filtered_indices), axis=1)
    df["predicted_cluster"] = labels
    return df



def trainManager(df,n_clusters, save_model=True):
    
    print("preprocessing starting now..")
    df = preprocess(df, train=True)
    
    print("training starting now..")

    coords = df[['longitude','latitude']]
    model_path = r"models/kmeans.pkl"
    labels, cluster_centres = train.main(coords,  n_clusters=n_clusters, return_labels=True,
                    save_model=save_model, model_path=model_path)

    print("model is trained")
    df["kmeans_cluster"] = labels

    cluster_df = getClusterdf(df, cluster_centres)
    return df, cluster_df



def breakLargeClusters(max_size, cluster_df, df):
    
    # cluster_df = pd.read_csv("cluster_dfs/model_clusters.csv", index_col='cluster')
    cluster_df.reset_index(inplace=True)

    break_these_clusters = cluster_df.loc[cluster_df.cluster_population>max_size]
    print(break_these_clusters)
    cluster_df = cluster_df.drop(index = break_these_clusters.index)
    print(cluster_df)
    if(len(cluster_df)==0):
        max_cluster = 0
    else:
        max_cluster = cluster_df.cluster.max() 
    last_cluster_number = max_cluster
    new_cluster_df = pd.DataFrame(columns = cluster_df.columns)
    new_cluster_df.index.name = 'cluster'

    iter = break_these_clusters.iterrows()
    while True:
        try:
           
            _, cluster = next(iter)
            cluster_number = cluster.cluster

            n = int(cluster.cluster_population/max_size)+1
            _, new_clusters = trainManager(df[df.kmeans_cluster==cluster_number], n, save_model=False)
            # print("here", len(new_clusters))
            new_clusters.reset_index(inplace=True)
            cluster_numbers = [cluster_number] 

            print("last", last_cluster_number)
            # print([i for i in range(1, len(new_clusters)+1)])
            cluster_numbers.extend([last_cluster_number+i for i in range(1, len(new_clusters))])

            print(cluster_numbers)
            new_clusters.cluster = cluster_numbers
            new_cluster_df = pd.concat((new_cluster_df,new_clusters), axis=0)
            last_cluster_number = max_cluster + len(new_cluster_df) - 1
            # return

        except StopIteration:
    
            # exception will happen when iteration will over
            print("new_clusters_made")
            break
    
    cluster_df = pd.concat((cluster_df,new_cluster_df), axis =0)
    cluster_df.set_index("cluster", drop=True, inplace=True)
    return cluster_df.sort_index(ascending=True)
