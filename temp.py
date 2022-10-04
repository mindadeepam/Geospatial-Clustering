
import pandas as pd

# cluster_df = pd.read_csv("cluster_dfs/model_clusters.csv", index_col="cluster")
# cluster_df

df = pd.read_csv("/Users/deepamminda/Documents/Geospatial-Clustering/sample_train_data.tsv", index_col=None, delimiter="\t")
df = df.drop(df[df.longitude==0.0].index)
df = df.loc[(df.longitude<=98) & (df.longitude>=60)]
df = df.loc[(df.latitude<=38) & (df.latitude>=8)]

df = df[["student_id", "latitude", "longitude"]]
df.to_csv("sample_test_data.csv")
# df.apply(axis=1, )