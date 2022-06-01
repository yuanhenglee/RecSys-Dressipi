import math
import random
import pandas as pd
from collections import defaultdict
from operator import itemgetter

# def LoadMovieLensData(filepath, train_rate):
#     ratings = pd.read_table(filepath, sep="::", header=None, names=["UserID", "MovieID", "Rating", "TimeStamp"],\
#                             engine='python')
#     ratings = ratings[['UserID','MovieID']]

#     train = []
#     test = []
#     random.seed(3)
#     for idx, row in ratings.iterrows():
#         user = int(row['UserID'])
#         item = int(row['MovieID'])
#         if random.random() < train_rate:
#             train.append([user, item])
#         else:
#             test.append([user, item])
#     return PreProcessData(train), PreProcessData(test)

def LoadData(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        return PreProcessData(lines)

def PreProcessData(originData):
    """
    建立User-Item表，结构如下：
        {"User1": {MovieID1, MoveID2, MoveID3,...}
         "User2": {MovieID12, MoveID5, MoveID8,...}
         ...
        }
    """
    trainData = dict()
    for i in range(1, len(originData)):
        line = originData[i].split(',')
        session_id = line[0]
        item_id = line[1]
        if( session_id == originData[i-1].split(',')[0]):
            session_id = int(session_id)
            item_id = int(item_id)
            trainData[session_id].add(item_id)
        else:
            session_id = int(session_id)
            item_id = int(item_id)
            trainData.setdefault(session_id, set())
            trainData[session_id].add(item_id)

    # for user, item in originData:
    #     trainData.setdefault(user, set())
    #     trainData[user].add(item)
    return trainData


class ItemCF(object):
    """ Item based Collaborative Filtering Algorithm Implementation"""
    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict() # 物品相似度矩阵

    def similarity(self):
        N = defaultdict(int) #记录每个物品的喜爱人数
        for user, items in self._trainData.items():
            #print(user, items)
            for i in items:
                self._itemSimMatrix.setdefault(i, dict())
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    self._itemSimMatrix[i].setdefault(j, 0)
                    if self._similarity == "cosine":
                        self._itemSimMatrix[i][j] += 1
                    elif self._similarity == "iuf":
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items) * 1.)
        for i, related_items in self._itemSimMatrix.items():
            for j, cij in related_items.items():
                self._itemSimMatrix[i][j] = cij / math.sqrt(N[i]*N[j])
        #print(self._itemSimMatrix)
        # 是否要标准化物品相似度矩阵
        if self._isNorm:

            for i, relations in self._itemSimMatrix.items():
                #print(i, relations)
                if( relations != {} ):
                    max_num = relations[max(relations, key=relations.get)]
                # 对字典进行归一化操作之后返回新的字典
                    self._itemSimMatrix[i] = {k : v/max_num for k, v in relations.items()}

    def recommend(self, user, N, K):
        """
        :param user: 被推荐的用户user
        :param N: 推荐的商品个数
        :param K: 查找的最相似的用户个数
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """
        recommends = dict()
        # 先获取user的喜爱物品列表
        items = self._trainData[user]
        # for i in self._trainData:
        #     print(self._trainData[i])
        for item in items:
            # 对每个用户喜爱物品在物品相似矩阵中找到与其最相似的K个
            for i, sim in sorted(self._itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                if i in items:
                    continue  # 如果与user喜爱的物品重复了，则直接跳过
                recommends.setdefault(i, 0.)
                recommends[i] += sim
        # 根据被推荐物品的相似度逆序排列，然后推荐前N个物品给到用户
        #return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])
        return sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N]

    def train(self):
        self.similarity()

if __name__ == "__main__":
    train = LoadData("./test_leaderboard_sessions.csv")
    #train, test = LoadMovieLensData("../Data/ml-1m/ratings.dat", 0.8)
    #print("train data size: %d, test data size: %d" % (len(train), len(test)))
    #print(train)
    ItemCF = ItemCF(train, similarity='iuf', norm=False)
    ItemCF.train()

    # 分别对以下4个用户进行物品推荐
    #print(ItemCF.recommend(1, 5, 80))
    fp = open("ans0601.csv", "w")
    fp.write("session_id,item_id,rank\n")
    for i in ItemCF._trainData:
        ans = ItemCF.recommend(i, 100, 2000)
        #print(ans)
        ii = 1
        for j,k in ans:
            #print(str(i)+','+str(j)+','+str(ii))
            fp.write(str(i)+','+str(j)+','+str(ii)+'\n')
            ii = ii+1
    # print(ItemCF.recommend(2, 5, 80))
    # print(ItemCF.recommend(3, 5, 80))
    # print(ItemCF.recommend(4, 5, 80))