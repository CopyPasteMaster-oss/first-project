# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   ML_project
# FileName:      valid_model.py
# ---------------------------------------------------------------------------
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report


# 模型
def valid_model():
    # 获取数据
    data = pd.read_csv("./data/test_set.csv")
    print(data.head())
    # 处理nan数据
    # print(data.isna().sum())
    data.dropna(inplace=True)
    # 确定特征和目标
    x = data['content']
    y = data['label']
    y_res = pd.get_dummies(y, dtype=int)['spam']
    # 加载词汇列表
    word_list = joblib.load("./data/word_list.pth")
    # 创建词频矩阵对象
    vector = CountVectorizer(vocabulary=word_list)
    # 创建词频矩阵
    x_res = vector.transform(x).toarray()
    # 加载模型
    model = joblib.load("./data/best_model.pth")
    # 模型评估
    y_pred = model.predict(x_res)
    print(classification_report(y_res, y_pred))

if __name__ == '__main__':
    valid_model()