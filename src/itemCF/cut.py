with open ('../../dataset/train_sessions.csv') as f:
    lines = f.readlines()
    fp = open("train_sessions_cutted.csv", "w")
    fp.write("session_id,item_id,date\n")
    for i in range(1, len(lines)):
        iniline = lines[i]
        line = lines[i].split(',')[2].split(' ')[0].split('-')
        # if (  (line[0] == '2021' and (line[1] == '05' or line[1] == '04' or line[1] == '03' ) or line[1] == '02' or line[1] == '01' ))   :
        if (  line[0] == '2021'  or  line[0] == '2020' and 6<int(line[1][1])<9 )   :
            fp.write(iniline)
