# generate df for training
# format: 
#   inner product 
# 1 
# 2
# .
# N

import argparse
import pickle
import pandas as pd
import numpy as np
import os
import sys
import csv
import time
from tqdm import tqdm
import gc
gc.enable()

np.set_printoptions(threshold=sys.maxsize)

n_features = 1

def cosine_similarity( v1, v2 ):
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def euclidean_similarity( v1, v2 ):
    return np.linalg.norm(v1-v2)

def inner_similarity( v1, v2 ):
    return np.dot(v1, v2)


def build_features( item_vector ):
    # for each candidate_item
    features_list = []
    for i in range(N):
        candidate_vector = vector_space[candidate_items[i]]
        features = []
        features.append(inner_similarity( item_vector, candidate_vector ))
        features_list.append(features)
    # print(feature_val)
    return features_list 


def combine_items_features( item_list ):
    vectors = [ vector_space[item_id] for item_id, date in item_list ]

    # simply sum up the vectors
    combined_vector = np.sum( vectors , axis=0 )
    return build_features( combined_vector )

# args
parser = argparse.ArgumentParser()
parser.add_argument("--session_path",
                    nargs='?',
                    help='session path',
                    default='dataset/train_sessions.pickle'
                    )
parser.add_argument("--purchase_path",
                    nargs='?',
                    help='purchase path',
                    default='dataset/train_purchases.pickle'
                    )
parser.add_argument("--vector_path",
                    nargs='?',
                    help='vector space path',
                    default='dataset/feature_vector_space.pickle'
                    )
parser.add_argument("--output_path",
                    nargs='?',
                    help='pickle path',
                    default='dataset/train_features/session_wp.csv'
                    )
parser.add_argument("--with_purchase", action="store_true")
# parser.add_argument("--by_session", action="store_true")
args = parser.parse_args()
try:
    session_path = args.session_path
    purchase_path = args.purchase_path
    vector_path = args.vector_path
    output_path = args.output_path
    with_purchase = args.with_purchase
    # by_session = args.by_session
except:
    raise "USAGE: python3 dump_embedding.py --feature_path ..."

# get data from pickle
try:
    with open(session_path, 'rb') as f:
        session_dict = pickle.load(f)
    with open(vector_path, 'rb') as f:
        vector_space = pickle.load(f)
    with open('./dataset/candidate_items.csv') as f:
        candidate_items = [ int(item) for item in f.read().split('\n') if item.isdigit() ]
        N = len(candidate_items)
    if with_purchase:
        with open(purchase_path, 'rb') as f:
            purchase_dict = pickle.load(f)
except:
    raise "Fail to load pickles."

if os.path.isfile( output_path ):
    print( output_path, "already exist, old file will replaced!!")
    os.remove( output_path )

# construct df
# for i, session_id in enumerate(list(session_dict.keys())[:100]):
start = 0
# end = 10000
end = len(session_dict) 
save_period = 1000

print( "Processing session", start , "to", end)
start_time = time.time()

features_lists = []
for i in tqdm(range(start, end)):
    session_id = list(session_dict.keys())[i]
    item_list = session_dict[session_id]
    features_list = combine_items_features( item_list )

    if with_purchase:
        purchase_id, purchase_date = purchase_dict[session_id]
        for j in range(N):
            features_list[j].append(1 if purchase_id == candidate_items[j] else 0)

    features_lists.extend(features_list)
    if i%save_period == 0 or i == end-1:
        with open( output_path, 'a') as f:
            wr = csv.writer(f)
            wr.writerows(features_lists)
            del features_lists
            gc.collect()
            features_lists = []

print("Done. Execution Time:", time.time() - start_time)
