# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:09:16 2021

@author: lijiaojiao
"""

import pandas as pd
import os
import re
import gensim
import nltk
import pyLDAvis


# 读取数据
papers = pd.read_csv("G:/SystemMine/ML/DataScience/ML/lda/data/papers.csv")

#删选有数据的列
paper = papers.iloc[:,0:7]


#######################1.数据清洗############

# 删除非文本的列，并取前100篇为例
paper = paper.drop(columns=['id', 'event_type', 'pdf_name'], axis=1).sample(100)

#去掉标点符号
paper['paper_text_processed'] = paper['paper_text'].map(lambda x: re.sub('[,\.!?]', '', str(x)))

#所有字符转化为小写
paper['paper_text_processed'].map(lambda x: x.lower())


########################2.做词云图############
from wordcloud import WordCloud
long_string = ','.join(list(paper['paper_text_processed'].values))

#创建词云对象
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

#生成词云图有
wordcloud.generate(long_string)

#可视化
wordcloud.to_image()


##设置停用词
from gensim.utils import simple_preprocess
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english') #设定语言
stop_words.extend(['from', 'subject', 're', 'edu', 'use']) #增加停用词


########################3.文本标记,去除停用词，构建词典############

#标记文本方法
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True 是删除标点符号 

# 去除停用词方法
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

#data_words_nostops = remove_stopwords(data_words)


#标记文本、去除停用词
data = paper.paper_text_processed.values.tolist()
data_words = list(sent_to_words(data))
data_words = remove_stopwords(data_words)


import gensim.corpora as corpora
id2word = corpora.Dictionary(data_words) #创建词典
#print("词典：",id2word.token2id)  result:词典： {'nan': 0, 'ability': 1, 'abstract': 2, 'access': 3, 'according': 4, 'achieve': 5,
texts = data_words 
corpus = [id2word.doc2bow(text) for text in texts] #基于词典id2word建立新的语料库
#print("语料库：",corpus)  result:[(67, 1), [(5, 4), (3, 3)......]  词67出现1次，词5出现4词



########################4.lda建模############

from pprint import pprint
# 设定主题数量
num_topics = 10
# 建立LDA模型
lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)   ##参数先默认
# 输出10个主题
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

##可视化

import pyLDAvis.gensim_models
import pickle 
# 可视化主题
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('G:/SystemMine/ML/DataScience/ML/lda/data/lda_'+str(num_topics))


if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        

with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
    
#保存为html网页
pyLDAvis.save_html(LDAvis_prepared, 'G:/SystemMine/ML/DataScience/ML/lda/data/lda_'+ str(num_topics) +'.html')
    
