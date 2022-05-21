#Usage: python3 MRR.py -a [purchases.csv] -p [prediction.csv]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--ans")
parser.add_argument("-p", "--prediction")
args = parser.parse_args()
MRR = 0
with open(args.ans) as f_ans:
    with open(args.prediction) as f_pre:
        lines = f_ans.readlines()
        pre_lines = f_pre.readlines()
        #len(lines)
        pre_no = 1
        for i in range(1,len(lines)):
            line = lines[i].split(',')
            session_id = line[0]
            item_id = line[1]
            #print(item_id)
            for j in range(100):
                pre_line = pre_lines[pre_no].split(',')
                pre_item_id = pre_line[1]
                if( item_id == pre_item_id ):
                    MRR = MRR + 1/(j+1)
                    #print(1/(j+1))
                #print(pre_line)
                pre_no = pre_no+1
            #print(line[0],line[1])
print("MRR = "+str(MRR))
