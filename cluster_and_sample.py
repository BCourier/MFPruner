# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:45:04 2025

@author: Wenhuan Song
"""

import pandas as pd
import numpy as np
import umap
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import distance
import os 

def calc_cluster_centers(df):
    '''
    calculate the center for each cluster.
    output is a dict where keys are cluster labels and values are cluster center coordinates.
    '''
    cluster_centers = {}
    for cluster in list(df["cluster"].unique()):
        df_tmp = df.loc[df["cluster"]==cluster]["latent"].apply(pd.Series)
        cluster_centers[cluster] = (df_tmp[0].mean(), df_tmp[1].mean())
    return cluster_centers


def result_plot(tags, mat, output_dir, palette=None):
    '''
    plot a dot figure to represent the result of clustering

        Input: 
            tags: list of tags for each vector
            mat: list of 2-dim vectors
    '''   
    
    temp = [[mat[i][0], mat[i][1], tags[i]] for i in range(len(tags))]
    xmin = min([tp[0] for tp in temp])
    ymin = min([tp[1] for tp in temp])
    xmax = max([tp[0] for tp in temp])
    ymax = max([tp[1] for tp in temp])
    df = pd.DataFrame(temp, columns=['x','y','cluster'])
    plt.figure(figsize=(4,4))
    sns.set(font_scale=1, style='ticks')
    savepath = output_dir + '\\plot.png'
    sns.scatterplot(data=df, x='x', y='y', hue='cluster',palette='bright', legend=False,
                  marker="+", s=120)
    plt.xlabel('')
    plt.ylabel('')
    plt.tick_params(axis='x', which='both', labelbottom=True)
    plt.tick_params(axis='y', which='both', labelleft=True)
    plt.xlim(xmin-1,xmax+1)
    plt.ylim(ymin-1,ymax+1)
    plt.savefig(savepath, dpi=600)
    # plt.show()
    
def calc_samples(vectors, array_name, npz_file_path, eps=0.8, sample_num=3, n_neighbors=40, min_dist=0, plot=True):

    reduced_vecs = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        n_jobs=1,
        random_state=42
        ).fit_transform(vectors)
    
    db = DBSCAN(eps=eps)
    db.fit(reduced_vecs)
    labels = list(db.labels_)
    
    df_result = pd.DataFrame()
    df_result["latent"] = reduced_vecs.tolist()
    df_result["cluster"] = labels
    df_result["sample_flag"] = False
    
    cluster_centers = calc_cluster_centers(df_result)
    df_result["dist"] = [distance.euclidean(cluster_centers[cluster], latent) 
                         for cluster, latent in zip(df_result["cluster"], df_result["latent"])]
    for group_name, group_data in df_result.groupby("cluster"):
        smallests = group_data.nsmallest(sample_num, "dist", keep="first")
        df_result.loc[smallests.index, "sample_flag"] = True
        
    df_cstat = pd.DataFrame()
    df_cstat[["cluster", "size"]] = [[k, v] for k, v in dict(Counter(labels)).items()]
    df_cstat["center"] = [cluster_centers[cluster] for cluster in df_cstat["cluster"]]
    
    output_dir = os.path.dirname(npz_file_path) + "\\" + array_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df_result.to_csv(output_dir + "\\sample_result.csv")
    df_cstat.to_csv(output_dir + "\\cluster_stats.csv")
    
    if plot:
        result_plot(labels, reduced_vecs.tolist(), output_dir)
        
    print("calculation finished, result outputed to "+output_dir)

# npz_file_path = '.\\test_file\\output.npz'
# npz_file = np.load(npz_file_path)
# data_arrays = npz_file[npz_file.files[0]]

# calc_samples(data_arrays, npz_file.files[0], npz_file_path)