import pandas as pd
from collections import defaultdict
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

"""用决策树预测获胜球队"""
dataset = pd.read_csv('sportsref_download (1).csv', parse_dates=["Date"], usecols=[0,1,2,3,4,5,6,9]) #read_csv()参数介绍：https://www.cnblogs.com/datablog/p/6127000.html
dataset.columns = ['Date', 'Score Type', 'Visitor Team', 'VisitorPts', 'Home Team', 'HomePts', 'OT?', 'Notes']  # 修改列名字
# # Note：筛选某一列名的所在行
# for i in range(len(dataset)):
#     if dataset['Score Type'][i] == '8:00p':
#         print(dataset[i:i+1])

# 主场胜就是1，否则客场胜为0
dataset['HomeWin'] = dataset["VisitorPts"] < dataset["HomePts"]
y_true = dataset["HomeWin"].values

# 新增2列值
dataset["HomeLastWin"] = False
dataset["VisitorLastWin"] = False
# print(dataset.ix[:5])
print("Home Win percentage: {0:.1f}%".format(100 * dataset["HomeWin"].sum() / dataset["HomeWin"].count()))

# 遍历所有数据，并设置主、客场上一场的胜负情况数据
won_last = defaultdict(int)
for index, row in dataset.iterrows():  # Note that this is not efficient
    home_team_name = row["Home Team"]
    visitor_team_name = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team_name]  # 初始化都是False
    row["VisitorLastWin"] = won_last[visitor_team_name]
    dataset.ix[index] = row
    # Set current win
    won_last[home_team_name] = row["HomeWin"]   # 每一次上次某行的主、客队伍胜负情况都会被当前的覆盖
    won_last[visitor_team_name] = not row["HomeWin"]
# print(dataset.ix[:5])


# ########### No1. 2列特征做训练
# 其取值不变时，用相同的训练集建树得到的结果一模一样，对测试集的预测结果也是一样的；其值改变时，得到的结果不同；若不设置此参数，则函数会自动选择一种随机模式，每次得到的结果也就不同。
clf = DecisionTreeClassifier(random_state=12)
X_previouswins = dataset[['HomeLastWin', 'VisitorLastWin']].values  # 把数据集中某几列凑成list单独拿出来
scores = cross_val_score(clf, X_previouswins, y_true, cv=10, scoring='accuracy')
print('主客队上场比赛结果为特征的Accuracy:{0:.1f}%'.format(np.mean(scores) * 100))


# ######## No2. 用2013年战绩来添加一个“主场对是否比对手水平高”的属性,3列特征做训练
standing = pd.read_csv("expend_standings.csv", skiprows=[0])  # 表名字不要
# print("取出某球队名字的行的所在'Team'列的值：", standing[standing['Team'] == 'Denver Nuggets']["Rk"].values[0])

dataset['HomeTeamRanksHigher'] = 0
for index, row in dataset.iterrows():
    home_team_name = row["Home Team"]
    visitor_team_name = row["Visitor Team"]
    if home_team_name == "New Orleans Pelicans":  # 检查名字是否更名
        home_team_name = "New Orleans Hornets"
    elif visitor_team_name == "New Orleans Pelicans":
        visitor_team_name = "New Orleans Hornets"
    home_rank = standing[standing['Team'] == home_team_name]["Rk"].values[0]
    visitor_rank = standing[standing['Team'] == visitor_team_name]["Rk"].values[0]
    row['HomeTeamRanksHigher'] = int(home_rank > visitor_rank)
    dataset.ix[index] = row
# print(dataset.ix[:5])


# No3.再添加一个新的特征：上一场比赛情况
X_homehigher = dataset[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_homehigher, y_true, scoring="accuracy")
print("添加比赛等级特征后的Accuracy:{0:.1f}".format(np.mean(scores) * 100))

last_math_winer = defaultdict(int)
dataset["HomeTeamWonLast"] = 0
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    teams = tuple(sorted([home_team, visitor_team]))  # 球队名字按照字母排序，上一场比赛的赢得为键
    row["HomeTeamWonLast"] = 1 if last_math_winer[teams] == row["Home Team"] else 0
    dataset.ix[index] = row
    winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
    last_math_winer[teams] = winner

X_lastwinner = dataset[["HomeTeamRanksHigher", "HomeTeamWonLast"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_lastwinner, y_true, scoring="accuracy")
print("根据上次比赛胜负和主队以往排名为特征的Accuracy:{0:.1f}%".format(np.mean(scores) * 100))

# No4.球队名字用独热码表示
encoding = LabelEncoder()
encoding.fit(dataset["Home Team"].values)
home_teams = encoding.transform(dataset["Home Team"].values)
visitor_teams = encoding.transform(dataset["Visitor Team"].values)
x_teams = np.vstack([home_teams, visitor_teams]).T
# print(x_teams)

onehot = OneHotEncoder()
x_teams_expand = onehot.fit_transform(x_teams).todense()
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, x_teams_expand, y_true, scoring="accuracy")
print("球队名字用独热码表示后：Accuracy:{0:.1f}%".format(np.mean(scores) * 100))

# ########################  随机森林 ####################################
clf = RandomForestClassifier(random_state=14)
score = cross_val_score(clf, x_teams_expand, y_true, scoring="accuracy")
print("随机森林做分类器后的Accuracy：{0:.1f}%".format(np.mean(score)*100))

# 多加入几个特征试试效果
x_all = np.hstack([X_homehigher, x_teams_expand])
print('测试特征有：', np.shape(x_all))
clf = RandomForestClassifier(random_state=14)
score = cross_val_score(clf, x_all, y_true, scoring="accuracy")
print("随机森林后的Accuracy：{0:.1f}%".format(np.mean(score)*100))

parameter_space = {
    "max_features": [2, 10, 'auto'],
    "n_estimators": [100],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [2, 4, 6]
                  }
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(x_all, y_true)
print("随机森林+网格搜索后的Accuracy：{0:.1f}%".format(grid.best_score_ * 100))
print(grid.best_params_)

# 参数最好的设置
clf = RandomForestClassifier(bootstrap=True,
                             random_state=14,
                             criterion="entropy",
                             max_depth=None,
                             max_features=2,
                             max_leaf_nodes=None,
                             min_samples_leaf=6,
                             min_samples_split=2,
                             n_estimators=100,
                             n_jobs=2,
                             oob_score=False,
                             verbose=0,
                             )
score = cross_val_score(clf, x_all, y_true, scoring="accuracy")
print("随机森林参数设置后的Accuracy：{0:.1f}%".format(np.mean(score)*100))
