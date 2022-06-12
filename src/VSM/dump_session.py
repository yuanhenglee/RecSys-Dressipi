# dump session 
# format: { session_id: [(item_id, date)], ... }

import argparse
import pickle
import datetime
import time

def is_date_selected( date_str ):
    date = datetime.datetime.strptime( date_str, "%Y-%m-%d %H:%M:%S")
    date_start = datetime.datetime(2021, 5, 1)

    if date > date_start:
        return True
    else:
        return False

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
            date = line.split(',')[2][:19]
            # sample only data after 2021
            if is_date_selected(date): 
                if session_id in session_dict:
                    session_dict[session_id].append( (item_id, date) )
                else:
                    session_dict[session_id] = [(item_id, date)]

    # sort session history by date
    for k, v in session_dict.items():
        session_dict[k] = sorted(v, key=lambda x: datetime.datetime.strptime(x[1], "%Y-%m-%d %H:%M:%S"), reverse=True)[:3]

    # print(list(session_dict.values())[:10])

    with open(output_path, 'wb') as f:
        pickle.dump(session_dict, f)
 #
if __name__ == "__main__":
    print("Building session dict...")
    start_time = time.time()
    main()
    print("Done. Execution Time:", time.time() - start_time)