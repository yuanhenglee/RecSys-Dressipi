# dump purchase
# format: { session_id:(item_id, date) } 

import argparse
import pickle

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--purchase_path",
                        nargs='?',
                        help='purchase path',
                        default='dataset/train_purchases.csv'
                        )
    parser.add_argument("--pickle_path",
                        nargs='?',
                        help='pickle path',
                        default='dataset/train_purchases.pickle'
                        )
    # parser.add_argument("--session", action = "store_true")
    # parser.add_argument("--item", action = "store_true")
    args = parser.parse_args()
    try:
        purchase_path = args.purchase_path
        pickle_path = args.pickle_path
        # use_item = args.item
        # use_session= args.session
    except:
        raise "USAGE: python3 dump_session.py --purchase_path ..."

    # get data into dict
    result = []
    purchase_dict = {}
    with open(purchase_path) as f:
        for line in f.readlines()[1:]:
            session_id = int(line.split(',')[0])
            item_id = int(line.split(',')[1])
            date = line.split(',')[2].strip()
            purchase_dict[session_id] = (item_id, date)

    # print(purchase_dict)

    with open(pickle_path, 'wb') as f:
        pickle.dump(purchase_dict, f)
 #
if __name__ == "__main__":
    main()