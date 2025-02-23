# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 01:17:03 2025

@author: Wenhuan Song
"""

import argparse
import cluster_and_sample
import numpy as np
import pandas as pd
import inflect


parser = argparse.ArgumentParser()
parser.add_argument('npz_file_path', type=str, help='the path of .npz format file, the clustering and sampling procedure will be conducted separately for each matrix in the file')
parser.add_argument('--eps', type=float, default=0.8, help='the min distance argument for DBSCAN, default is 0.8 in this tool')
parser.add_argument('--sample_num', type=int, default=3, help='the number of samples extracted for further validation from each cluster, default is 3')
parser.add_argument('--n_neighbors', type=float, default=40, help='argument for dim reduction function UMAP, see UMAP documents for further description. Default is 40')
parser.add_argument('--min_dist', type=float, default=0, help='argument for dim reduction function UMAP, see UMAP documents for further description. Default is 0')
parser.add_argument('--no-plot', action='store_true', help='call this argument to disable plotting clustering result')
args = parser.parse_args()

npz_file_path = args.npz_file_path
eps = args.eps
sample_num = args.sample_num
n_neighbors = args.n_neighbors
min_dist = args.min_dist
plot = bool(1-args.no_plot)
print(plot)

if eps < 0:
    raise ValueError("Invalid min distance for DBSCAN: ", eps)
if sample_num < 0:
    raise ValueError("Invalid number of samples: ", sample_num)
if n_neighbors < 0:
    raise ValueError("Invalid n_neighbors argument for UMAP: ", n_neighbors)
if min_dist < 0:
    raise ValueError("Invalid min_dist argument for UMAP: ", min_dist)

npz_file = np.load(npz_file_path)
data_arrays = npz_file.files

p = inflect.engine()
i = 1
for array in data_arrays:
    vectors = pd.DataFrame(npz_file[array])
    array_name = str(array)
    print("start processing "+p.ordinal(i)+f" matrix... ({array_name})")
    cluster_and_sample.calc_samples(vectors, array_name, npz_file_path, eps, sample_num, n_neighbors, min_dist, plot)
    i += 1 

print("all matrices processed. ")
    
    
    