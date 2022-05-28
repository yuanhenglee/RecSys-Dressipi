# dump session 
# format: { session_id: [(item_id, date)], ... }

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
    parser.add_argument("--output_path",
                        nargs='?',
                        help='output pickle path',
                        default='dataset/train_sessions.pickle'
                        )
    args = parser.parse_args()
    try:
        session_path = args.session_path
        output_path = args.output_path
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
            # sample only data after 2021
            if date.startswith("2021-04") or date.startswith("2021-05"):
                if session_id in session_dict:
                    session_dict[session_id].append( (item_id, date) )
                else:
                    session_dict[session_id] = [(item_id, date)]

    # print(session_dict)

    with open(output_path, 'wb') as f:
        pickle.dump(session_dict, f)
 #
if __name__ == "__main__":
    main()