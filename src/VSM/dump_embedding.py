# dump feature into pickle
# format: { item_id: TFIDFvector, ...}

import argparse
import pickle
import os
import time
from VectorSpace import VectorSpace

def vectorize( doc ):
    tokens = list(filter(None, doc.split(' ')))
    print(tokens)


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path",
                        nargs='?',
                        help='feature path',
                        default='dataset/feature_value_sep/'
                        )
    args = parser.parse_args()
    try:
        feature_path = args.feature_path
    except:
        raise "USAGE: python3 dump_embedding.py --feature_path ..."

    # get data into dict
    item_docs = {}
    item_vectors = {}
    # open all files in data_dir
    for root, dirs, files in os.walk(feature_path):
        for f_name in files[:]:
            item_id = int(f_name[:-4])
            f_path = os.path.join(root, f_name)
            # print(item_id, f_path)
            with open(f_path, 'r') as f:
                item_docs[item_id] = f.read()
                # item_vectors[item_id] = vectorize(f.read())
    
    # construct vector space
    start_time = time.time()
    print("Building VectorSpace...")
    vectorSpace = VectorSpace( item_docs )
    print("Done. Execution Time:", time.time() - start_time)

    result = vectorSpace.documentVectorsTFIDF

    pickle_path = 'dataset/feature_vector_space.pickle'

    with open(pickle_path, 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    main()