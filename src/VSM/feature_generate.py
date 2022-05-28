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

def cosine_similarity( v1, v2 ):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def euclidean_similarity( v1, v2 ):
    return np.linalg.norm(v1-v2)

def build_features( item ):
    ...

def combine_items_similarity( item_list ):
    # cal_item_similarity()
    v1 = vector_space[item1]
    v2 = vector_space[item2]

def main():
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
            print(candidate_items)
        if with_purchase:
            with open(purchase_path, 'rb') as f:
                purchase_dict = pickle.load(f)
    except:
        raise "Fail to load pickles."

    # construct df
    # for session_id, item_list in session_dict.items():
    #     if by_session:





    # with open(pickle_path, 'wb') as f:
    #     pickle.dump(result, f)

if __name__ == "__main__":
    main()