with open('../../dataset/test_leaderboard_sessions.csv') as f:
    lines = f.readlines()
    fp = open("./train_sessions_cutted.csv", "a")
    #fp.write("session_id,item_id,rank\n")
    for i in range(1, len(lines)):
        fp.write(lines[i])
