import pandas as pd
from sklearn import neighbors
#参考： https://www.jb51.net/article/60510.htm
# 1. https://mp.weixin.qq.com/s?__biz=MzU0MDQ1NjAzNg==&mid=2247486609&idx=1&sn=2865ab86b1ab0abf3e50e109bbf0ad14&chksm=fb39a99acc4e208c3bbcb410742adca7839408fc4bed9f4aa2446d4f500fcaa360e388e3c6c8&mpshare=1&scene=23&srcid=#rd
# 2. https://mp.weixin.qq.com/s?__biz=MzU0MDQ1NjAzNg==&mid=2247486622&idx=1&sn=d85a4866615f8a503151577abb28edcf&chksm=fb39a995cc4e20835fec4a9dcb0ec9b0047265e24bc013e11e69a7b8c259deab0dbbf0e4f303&mpshare=1&scene=23&srcid=#rd

# 读入外部数据
data3 = pd.read_excel(io=r'data3.xlsx')   # pandas 更新到0.24即可
'''
# 1.查看数据的情况
print(data3.shape)
print(data3.dtypes)
# 数值型转字符型
data3['id'] = data3['id'].astype(str)
# 字符型转数值型
data3['custom_amt'] = data3['custom_amt'].str[1:].astype(float)
# 字符型转日期型
data3['order_date'] = pd.to_datetime(data3['order_date'], format = '%Y年%m月%d日')

# 重新查看数据集的各变量类型
print(data3.dtypes)
print(data3.head())

# 判断数据中是否存在重复数据
print(data3.duplicated().any())

需要说明的是，在使用duplicated“方法”对数据行作重复性判断时，会返回一个与原数据行数相同的序列（如果数据行没有重复，则对应False，否则对应True），为了得到最终的判断结果，需要再使用any“方法”（即序列中只要存在一个True，则返回True）。

duplicated“方法”和drop_duplicates“方法”都有一个非常重要的参数，就是subset。
默认情况下不设置该参数时，表示对数据的所有列进行重复性判断；如果需要按指定的变量做数据的重复性判断时，
就可以使用该参数指定具体的变量列表。举例如下：
df = pd.DataFrame(dict(name = ['张三','李四','王二','张三','赵五','丁一','王二'],
                      gender = ['男','男','女','男','女','女','男'],
                      age = [29,25,27,29,21,22,27],
                      income = [15600,14000,18500,15600,10500,18000,13000],
                      edu = ['本科','本科','硕士','本科','大专','本科','硕士']))
# 默认情况下，对数据的所有变量进行判断
df.drop_duplicates()
print(df)
df.drop_duplicates(subset=['name','age'],inplace=True) # name、age一样就判为一样的数据
print(df,'\n####################################')
'''

'''2.缺失值的简单处理有：删除法、替换法和插补法'''
# 判断各变量中是否存在缺失值
print(data3.isnull().any(axis = 0))
# 各变量中缺失值的数量
print(data3.isnull().sum(axis = 0))  # axis为1代表统计行，0统计列
#缺失值比例
print(data3.isnull().sum(axis=0)/data3.shape[0])
# 判断数据行中是否存在缺失值
data3.isnull().any(axis = 1).any()
# 删除字段 -- 如删除缺失率非常高的edu特征
print(data3.drop(labels = 'edu', axis = 1, inplace=True))
# 数据预览
print(data3.head())
# 删除观测，-- 如删除age变量中所对应的缺失观测
data3_new = data3.drop(labels = data3.index[data3['age'].isnull()], axis = 0)
# 查看数据的规模
print(data3_new.shape)

# 替换法处理缺失值
data3.fillna(value = {'gender': data3['gender'].mode()[0], # 使用性别的众数替换缺失性别
                 'age':data3['age'].mean() # 使用年龄的平均值替换缺失年龄
                 },
          inplace = True # 原地修改数据
          )
# 再次查看各变量的缺失比例
print(data3.isnull().sum(axis = 0))

'''3.KNN插补缺失值：以年龄不缺失的其他5特征为训练模型，训练时拟合已有年龄的数据，然后预测（拟合）缺失部分的年龄。'''
# 读取数据
titanic = pd.read_csv('Titanic.csv')
# 删除缺失严重的Cabin变量
titanic.drop(labels='Cabin', axis = 1, inplace=True)
# 根据Embarked变量，删除对应的缺失行
titanic.dropna(subset=['Embarked'], inplace=True)

# 删除无关紧要的变量（这些变量对后面预测年龄没有太多的帮助）
titanic.drop(labels=['PassengerId','Name','Ticket','Embarked'], axis = 1, inplace=True)
# 将字符型的性别变量映射为数值变量
titanic.Sex = titanic.Sex.map({'male':1, 'female':0})

# 将数据拆分为两组，一是年龄缺失组，二是年龄非缺失组，后续基于非缺失值构建KNN模型，再对缺失组做预测
nomissing = titanic.loc[~titanic.Age.isnull(),]
missing = titanic.loc[titanic.Age.isnull(),]
# X是除了age的所有列的index，即属性名字
X = nomissing.columns[nomissing.columns != 'Age']
# 构建模型
knn = neighbors.KNeighborsRegressor()
print('\n===========>', nomissing[X].isnull().any(axis=1).any(), nomissing[X].isnull().any,nomissing[X].isnull().sum(axis=0))
# 模型拟合
knn.fit(nomissing[X], nomissing.Age)
# 缺失表的年龄预测
pred_age = knn.predict(missing[X])
print(pred_age)

# Titanic预测kaggle竞赛： https://www.jianshu.com/p/9b6ee1fb7a60?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation