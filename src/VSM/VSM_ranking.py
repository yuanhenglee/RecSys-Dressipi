import csv
from tqdm import tqdm

session_ids = []
with open('dataset/test_leaderboard_sessions.csv') as f:
    for line in f.readlines()[1:]:
        # if len(line):
        session_id = int(line.split(',')[0])
        if len(session_ids) == 0 or session_ids[-1] != session_id:
            session_ids.append(session_id)

# print(session_ids[:10])

result = []
for i in tqdm(range(len(session_ids))):
    session_id = session_ids[i]
    data = []
    with open('dataset/test_features/test_s_' + str(session_id)) as f:
        data = list(csv.reader(f, delimiter=','))[1:]

    data.sort(key=lambda x: float(x[1]), reverse=True)       

    # print(data[:10])
    for i in range(100):
        result.append(str(session_id) + ',' + data[i][0] + ',' + str(i+1) )

# print(result)

with open( 'result/VSM_ranking_LB.csv', 'w') as f:
  f.write("session_id,item_id,rank\n")
  f.write('\n'.join(result))