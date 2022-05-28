# generate df for training
# format: 
#   cos euc
# 1 
# 2
# .
# N

import argparse
import pickle
import pandas as pd
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

n_features = 2

def cosine_similarity( v1, v2 ):
    return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def euclidean_similarity( v1, v2 ):
    return np.linalg.norm(v1-v2)

def build_features( item_id, purchase_item_id = 0 ):
    item_vector = vector_space[item_id]
    feature_val = np.zeros( shape = (N,n_features+1) )
    for i in range(N):
        candidate_vector = vector_space[candidate_items[i]]
        feature_val[i][0] = cosine_similarity( item_vector, candidate_vector )
        feature_val[i][1] = euclidean_similarity( item_vector, candidate_vector )
        feature_val[i][n_features] = 1 if candidate_items[i] == purchase_item_id else 0
    # print(feature_val)
    return feature_val


def combine_items_features( item_list, purchase_item_id = 0 ):
    feature_val = np.zeros( shape= (N,n_features+1) )
    for item_id, date in item_list:
        feature_val += build_features( item_id, purchase_item_id )
    return feature_val


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
                    default='dataset/train_sessions.pickle'
                    )
parser.add_argument("--with_purchase", action="store_true")
parser.add_argument("--by_session", action="store_true")
args = parser.parse_args()
try:
    session_path = args.session_path
    purchase_path = args.purchase_path
    vector_path = args.vector_path
    output_path = args.output_path
    with_purchase = args.with_purchase
    by_session = args.by_session
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

# construct df
feature_val = []
# i = 0
for session_id, item_list in session_dict.items():
    if by_session:
        if with_purchase:
            print(purchase_dict[session_id])
            feature_val.append(combine_items_features( item_list, purchase_dict[session_id][0] ))
        else:
            feature_val.append(combine_items_features( item_list ))

        # i+=1
        # if i >10:
        #     break

feature_val = np.vstack(feature_val)
print(feature_val)


with open(output_path, 'wb') as f:
    pickle.dump(feature_val, f)