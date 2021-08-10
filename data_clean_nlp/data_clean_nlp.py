# -*- coding: utf-8 -*-
"""
The review of restaurant 

@author: lijiaojiao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #不设置quoting，默认会去除英文双引号，设置quoting = 3，会原样读取内容，包括引号。

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  #词根化

"""
# =============================================================================
# 首先对其中一行进行处理
# #去除所有标点符合
# review = re.sub("[^a-zA-Z]"," ", dataset["Review"][0]) 
# #大写转化为小写
# review = review.lower()
# # 以空格为分隔符进行分词
# review = review.split()
# #去除停用词，set(stopwords.words('english')，设置为set，提升处理速度
# review = [word for word in review if not word in set(stopwords.words('english'))]
# #去除停用词并词根化处理
# ps = PorterStemmer()
# review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
# # 每个单词串起来，组成新的字符串
# review = " ".join(review)
# =============================================================================
"""

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#创建词袋模型
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()  #矩阵
y = dataset.iloc[:, 1].values

# 训练集和测试集的划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# 用贝叶斯训练数据集
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


#预测
y_pred = classifier.predict(X_test)

#计算混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)