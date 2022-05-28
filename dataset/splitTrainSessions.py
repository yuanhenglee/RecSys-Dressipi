import argparse


def splitData(year, month):

    with open('train_sessions.csv', 'r') as f:
        lines = f.readlines()

        if month == None:
            output_file = open(f'./{year}.csv', 'w')
            output_file.write('session_id,item_id,date\n')

            for line in lines:
                if f',{year}-' in line:
                    output_file.write(line)

            output_file.close()

        else:
            output_file = open(f'./{year}-{month}.csv', 'w')
            output_file.write('session_id,item_id,date\n')

            for line in lines:
                if f',{year}-{month}-' in line:
                    output_file.write(line)

            output_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-y')
    parser.add_argument('-m')

    args = parser.parse_args()

    year = args.y
    month = args.m

    if year == None:
        print(
            'year can not be null\nexample: python3 splitTrainSessions.py -y 2011 -m 04')
        exit()

    splitData(year, month)
