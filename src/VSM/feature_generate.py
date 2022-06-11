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
from numpy import zeros
print("can't use cupy..")
from numpy import dot
from numpy.linalg import norm
from numpy import argsort
import csv
import time
from tqdm import tqdm
import gc
from datetime import datetime

# setting param
n_train_sample = 500 # top 150 inner product samples
pickle_protocol = 5

feature_cols = [
    'candidate_item_id',
    'top1_inner',
    'mean_top3_inner',
    'max_top3_inner',
    'top1_cos',
    'mean_top3_cos',
    'max_top3_cos'
    # 'top1_itemCF'
    # 'mean_top3_itemCF',
    # 'max_top3_itemCF',
    ]

def cosine_similarity(v1, v2, inner):
    return float(inner/(norm(v1)*norm(v2)))

def inner_similarity(v1, v2):
    return dot(v1, v2)

def select_top_inner( item_vector, purchase_id ):
    # cal all N inner 
    N_inner = zeros(N)
    for i in range(N):
        candidate_vector = vector_space[candidate_items[i]]
        N_inner[i] = inner_similarity(item_vector, candidate_vector)
    
    selected_ids = [ candidate_items[int(i)] for i in argsort(N_inner)[-1*n_train_sample:]]

    if purchase_id in selected_ids or purchase_id not in candidate_items:
        return selected_ids
    else:
        return [purchase_id] + selected_ids[1:]


def build_features( item_vectors, session_id ):
    y = None 
    if with_purchase:
        purchase_id, _ = purchase_dict[session_id]
        selected_ids = select_top_inner( item_vectors[0], purchase_id )
        y = [0] * n_train_sample
    else:
        # selected_ids = candidate_items
        selected_ids = select_top_inner( item_vectors[0], -1)
    # for each candidate_item
    features_list = zeros((len(selected_ids), len(feature_cols)))

    # item dic for recording item_id, inner product
    # item_dic = {}  # //
    for i in range(len(selected_ids)):
        candidate_item = selected_ids[i]
        candidate_vector = vector_space[candidate_item]

        top_items_inner = [inner_similarity(item_vector, candidate_vector) for item_vector in item_vectors]
        top_items_cosine= [cosine_similarity(item_vectors[j], candidate_vector, top_items_inner[j]) for j in range(len(item_vectors))]
        # top_items_itemCF= [inner_similarity(item_vector, candidate_vector) for item_vector in item_vectors]

        # item_id label, for easier sampling later
        features_list[i][0] = candidate_item

        if with_purchase and candidate_item == purchase_id:
            y[i] = 1

        features_list[i][1] = top_items_inner[0]
        features_list[i][2] = np.mean(top_items_inner)
        features_list[i][3] = np.max(top_items_inner)

        features_list[i][4] = top_items_cosine[0]
        features_list[i][5] = np.mean(top_items_cosine)
        features_list[i][6] = np.max(top_items_cosine)

        # features_list[i][7] = top_items_cosine[0]
        # features_list[i][8] = np.mean(top_items_cosine)
        # features_list[i][9] = np.max(top_items_cosine)


    return features_list, y


def combine_items_features( session_id ):
    item_list = session_dict[session_id]

    top_item_ids = [item_id for item_id, _ in item_list[:3]]
    top_item_vectors = [vector_space[item_id] for item_id in top_item_ids]

    return build_features( top_item_vectors, session_id )


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
                    default='dataset/train_features/train'
                    )
parser.add_argument("--with_purchase", action="store_true")
parser.add_argument("--as_pickle", action="store_true")
parser.add_argument("--as_csv", action="store_true")
args = parser.parse_args()
try:
    session_path = args.session_path
    purchase_path = args.purchase_path
    vector_path = args.vector_path
    output_path = args.output_path
    with_purchase = args.with_purchase
    as_pickle = args.as_pickle
    as_csv = args.as_csv
except:
    raise "USAGE: python3 dump_embedding.py --feature_path ..."

# get data from pickle
try:
    with open(session_path, 'rb') as f:
        session_dict = pickle.load(f)
    with open(vector_path, 'rb') as f:
        vector_space = pickle.load(f)
    with open('./dataset/candidate_items.csv') as f:
        candidate_items = [int(item)
                           for item in f.read().split('\n') if item.isdigit()]
        N = len(candidate_items)
    if with_purchase:
        with open(purchase_path, 'rb') as f:
            purchase_dict = pickle.load(f)
except:
    raise "Fail to load pickles."

start = 0
save_period = 10000
end = 1000
# end = len(session_dict)
# construct df

print("Processing session", start, "to", end)
start_time = time.time()

features_lists = []
y_lists = []
for i in tqdm(range(start, end)):
    session_id = list(session_dict.keys())[i]
    X, y = combine_items_features(session_id)  # //

    # training data with purchase
    if with_purchase:
        features_lists.extend(X)
        y_lists.extend(y)

        if i % save_period == save_period-1 or i == end-1:
            if as_pickle:
                with open( output_path + '_' + 'X' + '_' + str(i//save_period) + '.pickle', 'wb' ) as f:
                    pickle.dump( features_lists, f, pickle_protocol )
                with open( output_path + '_' + 'y' + '_' + str(i//save_period) + '.pickle', 'wb' ) as f:
                    pickle.dump( y_lists, f, pickle_protocol )
            if as_csv:
                with open(output_path + '_' + 'X' + '_' + str(i//save_period) + '.csv', 'w') as f:
                    wr = csv.writer(f)
                    wr.writerow( feature_cols )
                    wr.writerows(features_lists)
                with open(output_path + '_' + 'y' + '_' + str(i//save_period) + '.csv', 'w') as f:
                    f.write('purchased\n')
                    for y_int in y_lists:
                        f.write(str(y_int))

            del features_lists
            del y_lists
            gc.collect()
            features_lists = []
            y_lists = []
# testing data without purchase
    else:
        if as_pickle:
            with open( output_path + '_' + 'X' + '_' + str(session_id) + '.pickle', 'wb' ) as f:
                pickle.dump( X, f , pickle_protocol)
        if as_csv:
            with open(output_path + '_' + 'X' + '_' + str(session_id) + '.csv', 'w') as f:
                wr = csv.writer(f)
                wr.writerow( feature_cols )
                wr.writerows(X)

print("Done. Execution Time:", time.time() - start_time)
