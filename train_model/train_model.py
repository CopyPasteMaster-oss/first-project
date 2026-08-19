# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   ML_project
# FileName:      train_model.py
# ---------------------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def train_model():
    # 获取数据
    data = pd.read_csv('./data/train_set.csv')
    # 处理nan数据
    # print(data.isna().sum())
    data.dropna(inplace=True)
    # 数据的基本处理
    x = data['content']
    y = data['label']
    y_res = pd.get_dummies(y, dtype=int)['spam']   # 1: spam, 0: ham
    # 特征工程
    vector = CountVectorizer(max_features=6000)     # max_features: 设置最大的特征数量
    # vector = CountVectorizer()
    # 生成词频矩阵
    x_res = vector.fit_transform(x).toarray()
    print(x_res.shape)
    # 训练模型
    model = MultinomialNB()
    model.fit(x_res, y_res)
    # 存储模型、词汇列表
    joblib.dump(model, './data/best_model.pth')
    joblib.dump(vector.get_feature_names_out(), './data/word_list.pth')


if __name__ == '__main__':
    train_model()