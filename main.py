import pandas as pd
import argparse
from utils import preprocess, getClusterdf, getClusters, breakLargeClusters, trainManager
import train
import time

# df = pd.read_csv('date_and_null_managed.csv')  #13L
# df

def main(params):
    action = params.action
    path_to_data = params.path_to_data
    n_clusters = params.n_clusters
    max_size = params.max_size

    if action=="none":
        print(action, path_to_data, n_clusters, type(n_clusters), max_size)
        return

    df = pd.read_csv(path_to_data, delimiter="\t")
    # df = pd.read_csv(path_to_data)
    
    

    if action=="train":
        
        df, cluster_df = trainManager(df, n_clusters)
        print(f"1st iteration of {n_clusters} clusters made. cluster_df saved")

        print("breaking larger clusters now..")
        if(max_size>0):
            cluster_df.to_csv("cluster_dfs/org_model_clusters.csv")
            cluster_df = breakLargeClusters(max_size, cluster_df, df)
            cluster_df.to_csv("cluster_dfs/constrained_clusters.csv")
        else:
            cluster_df.to_csv("cluster_dfs/model_clusters.csv")

        # print(cluster_df)
        print("training finished. cluster_df saved")


    if action=="predict":
        st = time.time()
        cluster_df = pd.read_csv("./cluster_dfs/model_clusters.csv")
        df = preprocess(df, predict=True)

        df = getClusters(df, cluster_df[["center_lon", "center_lat"]])

        df.to_csv("results.csv")
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
         
    pass 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='use kmeans to predict clusters using trained model or train new model')
    
    parser.add_argument('-a', '--action', help='action to perform. ie train/predict')
    parser.add_argument('--path_to_data', help='path to data file (csv) for action')
    parser.add_argument('--n_clusters', help='number of clusters required', type=int, default=1000)
    parser.add_argument('--max_size', help='max_size of clusters', type=int, default=0)
    # parser.add_argument('--db', help='db name for postgres')
    # parser.add_argument('--table_name', help='table name to insert data into')
    # parser.add_argument('--url', help='url of the csv file')


    args = parser.parse_args()
    main(args)