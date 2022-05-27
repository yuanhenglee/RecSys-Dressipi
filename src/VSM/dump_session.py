# dump session 
# 1. session:  
# 2. item: 

import argparse
import pickle

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_path",
                        nargs='?',
                        help='session path',
                        default='dataset/train_sessions.csv'
                        )
    parser.add_argument("--pickle_path",
                        nargs='?',
                        help='pickle path',
                        default='dataset/train_sessions.pickle'
                        )
    # parser.add_argument("--session", action = "store_true")
    # parser.add_argument("--item", action = "store_true")
    args = parser.parse_args()
    try:
        session_path = args.session_path
        pickle_path = args.pickle_path
        # use_item = args.item
        # use_session= args.session
    except:
        raise "USAGE: python3 dump_session.py --session_path ..."

    # get data into dict
    result = []
    session_dict = {}
    with open(session_path) as f:
        for line in f.readlines()[1:]:
            session_id = int(line.split(',')[0])
            item_id = int(line.split(',')[1])
            date = line.split(',')[2].strip()
            if session_id in session_dict:
                session_dict[session_id].append( (item_id, date) )
            else:
                session_dict[session_id] = [(item_id, date)]

    # print(session_dict)

    with open(pickle_path, 'wb') as f:
        pickle.dump(session_dict, f)
 #
if __name__ == "__main__":
    main()