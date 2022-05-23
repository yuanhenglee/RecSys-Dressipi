import argparse
import os
from VectorSpace import VectorSpace
import time
import pandas as pd

documents = []
document_ids = []
method_name = {
    1:"TF Weighting + Cosine Similarity",
    2:"TF Weighting + Euclidean Distance",
    3:"TF-IDF Weighting + Cosine Similarity",
    4:"TF-IDF Weighting + Euclidean Distance"
}

candidate_items = set( int(item_id) for item_id in open("dataset/candidate_items.csv").readlines()[1:])
# print(candidate_items)

# print result in nice format
def run_method( session_id, item_ids, method, feedback = False):
    query = ''
    for item_id in item_ids:
        with open(data_dir + item_id + '.txt') as f:
            query += (f.read() + ' ')
    
    # print("\n\n",query)
    # timing each search
    # start_time = time.time()

    name = method_name[method]
    if feedback:
        name += ' with feedback'
    rating = vectorSpace.search(query, method, feedback)
    ranking = [(document_ids[i][:-4], rating[i]) for i in range(len(rating))]
    ranking = sorted( ranking, key = lambda x: x[1], reverse = method%2)


    i = 0
    n_retrieved_item = 0
    while n_retrieved_item < 100 and i < len(ranking):
        if int(ranking[i][0]) in candidate_items:
            # print("check", ranking[i][0])
            n_retrieved_item += 1
            print(str(session_id) + ',' + str(ranking[i][0]) +',' + str(ranking[i][1]))
            prediction_file.write( str(session_id) + ',' + str(ranking[i][0]) + ',' + str(n_retrieved_item) + '\n')
        i+=1
    
    # print("Sessions:", str(session_id), "\nExecution Time:", time.time() - start_time)


if __name__ == "__main__":

    # handle args
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",
                        nargs='?',
                        help='type of operation',
                        default='0'
                        )
    # parser.add_argument('--feedback', action='store_true')
    # parser.add_argument('--chinese', action='store_true')
    parser.add_argument('--X')
    args = parser.parse_args()

    # feedback = args.feedback
    # chinese = args.chinese
    X_path = args.X
    method = int(args.method)
    tf_only = method in [1,2]
    # query = []
    # for q in args.query:
    #     query.extend(q.split(' '))

    # select data
    data_dir = 'dataset/feature_value/'

    # open all files in data_dir
    for root, dirs, files in os.walk(data_dir):
        for f_name in files[:]:
            f_path = os.path.join(root, f_name)
            if int(f_name[:-4]) in candidate_items:
                document_ids.append(f_name)
                with open(f_path, 'r') as f:
                    doc_str = f.read()
                    documents.append(doc_str)


    # construct vector space
    start_time = time.time()
    print("Building VectorSpace...")
    vectorSpace = VectorSpace( documents, tf_only = tf_only, chinese = False )
    print("Done. Execution Time:", time.time() - start_time)

    start_time = time.time()
    print("Query...")
    # extract query words
    sessions = {}
    with open(X_path) as f:
        lines = f.readlines()
    for line in lines[1:]:
        session_id = line.split(',')[0]
        item_id = line.split(',')[1]
        if session_id not in sessions:
            sessions[session_id] = [str(item_id)]
        else:
            sessions[session_id].append(str(item_id))
    
    # print(sessions)

    prediction_file = open('./prediction.csv', 'w')
    prediction_file.write('session_id,item_id,rank\n')

    for session_id, item_ids in sessions.items():
        run_method(session_id, item_ids, method)

    prediction_file.close()
    print("Done. Execution Time:", time.time() - start_time)