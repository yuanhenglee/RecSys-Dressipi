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
from datetime import datetime

gc.enable()

np.set_printoptions(threshold=sys.maxsize)

# setting param
n_train_sample = 150 # top 150 inner product samples
pickle_protocol = 5

feature_cols = [
    'candidate_item_id',
    'top1_inner',
    'mean_top3_inner',
    'max_top3_inner',
    # 'top1_cos',
    # 'mean_top3_cos',
    # 'max_top3_cos'
    ]

def cosine_similarity(v1, v2, inner):
    return inner/(np.linalg.norm(v1)*np.linalg.norm(v2))

def inner_similarity(v1, v2):
    return np.dot(v1, v2)

def select_top_inner( item_vector, purchase_id ):
    # cal all N inner 
    N_inner = np.zeros(N)
    for i in range(N):
        candidate_vector = vector_space[candidate_items[i]]
        N_inner[i] = inner_similarity(item_vector, candidate_vector)
    
    selected_ids = [ candidate_items[i] for i in np.argsort(N_inner)[-1*n_train_sample:]]

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
        selected_ids = candidate_items
    # for each candidate_item
    features_list = np.zeros((n_train_sample, len(feature_cols)))

    # item dic for recording item_id, inner product
    # item_dic = {}  # //
    for i in range(n_train_sample):
        candidate_item = selected_ids[i]
        candidate_vector = vector_space[candidate_item]

        top_items_inner = [inner_similarity(item_vector, candidate_vector) for item_vector in item_vectors]
        # top_items_cosine= [cosine_similarity(item_vectors[i], candidate_vector, top_items_inner[i]) for i in range(len(item_vectors))]

        # item_id label, for easier sampling later
        features_list[i][0] = candidate_item

        if with_purchase and candidate_item == purchase_id:
            y[i] = 1

        features_list[i][1] = top_items_inner[0]
        features_list[i][2] = np.mean(top_items_inner)
        features_list[i][3] = np.max(top_items_inner)
        # features_list[i][4] = top_items_cosine[0]
        # features_list[i][5] = np.mean(top_items_cosine)
        # features_list[i][6] = np.max(top_items_cosine)

        # item_dic[i] = (top_items_inner[0])  # //

    # sort_dic = {k: v for k, v in sorted(
    #     item_dic.items(), key=lambda item: item[1], reverse=True)}  # //

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
args = parser.parse_args()
try:
    session_path = args.session_path
    purchase_path = args.purchase_path
    vector_path = args.vector_path
    output_path = args.output_path
    with_purchase = args.with_purchase
    as_pickle = args.as_pickle
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
# end = 5
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
    #     purchase_id, purchase_date = purchase_dict[session_id]
    #     y = [ 1 if purchase_id == candidate_item else 0 for candidate_item in candidate_items ]

    #     # only keep first 100 inner product value
    #     sort_list = []
    #     y_sort_list = []
    #     num = 0
    #     flag = False
    #     for k in item_dic.keys():
    #         if num >= n_train_sample and flag == True:
    #             break
    #         num += 1

    #         if num >= n_train_sample and flag == False:
    #             # if features_list[k][3] == 1:
    #             if y[k] == 1:
    #                 sort_list.append(features_list[k])
    #                 y_sort_list.append([y[k]])
    #                 break
    #             else:
    #                 continue

    #         sort_list.append(features_list[k])
    #         y_sort_list.append([y[k]])

    #         if y[k] == 1:
    #         # if features_list[k][3] == 1:
    #             flag = True

    #     features_lists.extend(sort_list)
    #     y_lists.extend(y_sort_list)

    # # //
        features_lists.extend(X)
        y_lists.extend(y)

        if i % save_period == save_period-1 or i == end-1:
            if as_pickle:
                with open( output_path + '_' + 'X' + '_' + str(i//save_period) + '.pickle', 'wb' ) as f:
                    pickle.dump( features_lists, f, pickle_protocol )
                with open( output_path + '_' + 'y' + '_' + str(i//save_period) + '.pickle', 'wb' ) as f:
                    pickle.dump( y_lists, f, pickle_protocol )
            else:
                with open(output_path + '_' + 'X' + '_' + str(i//save_period) + '.csv', 'w') as f:
                    wr = csv.writer(f)
                    wr.writerow( feature_cols )
                    wr.writerows(features_lists)
                with open(output_path + '_' + 'y' + '_' + str(i//save_period) + '.csv', 'w') as f:
                    f.write('purchased\n')
                    f.write('\n'.join(y_lists))

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
        else:
            with open(output_path + '_' + 'X' + '_' + str(session_id) + '.csv', 'w') as f:
                wr = csv.writer(f)
                wr.writerow( feature_cols )
                wr.writerows(X)

print("Done. Execution Time:", time.time() - start_time)
