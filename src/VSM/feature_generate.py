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
    'max_top3_cos',
    'top1_itemCF',
    'mean_top3_itemCF',
    'max_top3_itemCF',
    'agg1.5_itemCF',
    'top1_popularity_cur_summer',
    'top1_popularity_cur_spring',
    'top1_popularity_last_summer',
    'can_popularity_cur_summer',
    'can_popularity_cur_spring',
    'can_popularity_last_summer',
    ]

itemCF_weight = np.arange(1, 31, 1.5)

def cosine_similarity(v1, v2, inner):
    return float(inner/(norm(v1)*norm(v2)))

def inner_similarity(v1, v2):
    return dot(v1, v2)

def itemCF_similarity( item1, item2 ):
    try:
        return itemCF_matrix[item1][item2]
    except KeyError:
        return 0

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

def select_top_itemCF( item_id, purchase_id ):
    # cal all N inner 
    N_itemCF = zeros(N)
    for i in range(N):
        N_itemCF[i] = itemCF_similarity( item_id, candidate_items[i])

    selected_ids = [ candidate_items[int(i)] for i in argsort(N_itemCF)[-1*n_train_sample:]]

    if purchase_id in selected_ids or purchase_id not in candidate_items:
        return selected_ids
    else:
        return [purchase_id] + selected_ids[1:]

def build_features( item_ids, session_id ):

    item_vectors = [vector_space[item_id] for item_id in item_ids]

    y = [0] * n_train_sample
    if with_purchase:
        purchase_id, _ = purchase_dict[session_id]
        # selected_ids = select_top_inner( item_vectors[0], purchase_id )
        selected_ids = select_top_itemCF( item_ids[0], purchase_id )
    else:
        # selected_ids = select_top_inner( item_vectors[0], -1)
        selected_ids = select_top_itemCF( item_ids[0], -1)

    features_list = zeros((len(selected_ids), len(feature_cols)))

    for i in range(len(selected_ids)):
        candidate_item = selected_ids[i]
        candidate_vector = vector_space[candidate_item]

        top_items_inner = [inner_similarity(item_vector, candidate_vector) for item_vector in item_vectors[:3]]
        top_items_cosine= [cosine_similarity(item_vectors[j], candidate_vector, top_items_inner[j]) for j in range(len(item_vectors[:3]))]
        top_items_itemCF= [itemCF_similarity(item_id, candidate_item) for item_id in item_ids]

        # item_id label, for easier sampling later
        features_list[i][0] = candidate_item

        if with_purchase:
            if candidate_item == purchase_id:
                y[i] = 1
            elif purchase_id in item_ids:
                y[i] = 0.2

        features_list[i][1] = top_items_inner[0]
        features_list[i][2] = np.mean(top_items_inner)
        features_list[i][3] = np.max(top_items_inner)

        features_list[i][4] = top_items_cosine[0]
        features_list[i][5] = np.mean(top_items_cosine)
        features_list[i][6] = np.max(top_items_cosine)

        features_list[i][7] = top_items_itemCF[0]
        features_list[i][8] = np.mean(top_items_itemCF[:3])
        features_list[i][9] = np.max(top_items_itemCF[:3])

        features_list[i][10] = np.sum(itemCF_weight[:len(item_ids)] * top_items_itemCF[::-1])

        features_list[i][11] = popularity_cur_summer[item_ids[0]]
        features_list[i][12] = popularity_cur_spring[item_ids[0]]
        features_list[i][13] = popularity_last_summer[item_ids[0]]

        features_list[i][14] = popularity_cur_summer[candidate_item]
        features_list[i][15] = popularity_cur_spring[candidate_item]
        features_list[i][16] = popularity_last_summer[candidate_item]

    return features_list, y


def combine_items_features( session_id ):
    item_list = session_dict[session_id]

    top_item_ids = [item_id for item_id, _ in item_list]

    return build_features( top_item_ids, session_id )


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
    with open('./src/itemCF/itemSimMatrix.pickle', 'rb') as f:
        itemCF_matrix = pickle.load(f)

    popularity_df = pd.read_csv('./src/VSM/popular_train_can.csv', index_col = 'item_id')
    popularity_cur_summer = popularity_df[['2021-05', '2021-06']].sum(axis =1).copy()
    popularity_cur_spring = popularity_df[['2021-02', '2021-03', '2021-04']].sum(axis =1).copy()
    popularity_last_summer = popularity_df[['2020-05', '2020-06', '2020-07']].sum(axis =1).copy()
    del popularity_df

    if with_purchase:
        with open(purchase_path, 'rb') as f:
            purchase_dict = pickle.load(f)
except:
    raise "Fail to load files."

start = 0
save_period = 10000
# end = 1000
end = len(session_dict)
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
                        f.write(str(y_int) + '\n')

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
