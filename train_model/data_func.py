# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   ML_project
# FileName:      data_func.py
# ---------------------------------------------------------------------------
import re
import jieba.posseg
import zhconv
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ML_project1.utils.data_pro import data_clean, word_cut

# 根路径
base_dir = r"D:\学习\2026-08-17_模型训练和部署\doc\data\trec06c"

# index文件路径
path1 = r"D:\学习\2026-08-17_模型训练和部署\doc\data\trec06c\full\index"

# 数据处理
def data_get(path=path1):
    # 读取index文件
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    # 定义2个列表，分别存放各邮件的label和处理之后的邮件内容
    contents = []
    labels = []
    # 创建进度条
    tq = tqdm(range(len(lines)), desc="数据清洗")
    # 获取index文件中每行的内容
    for line in lines:
        # 邮件的标签
        label = line.strip().split(" ..")[0]
        # 邮件正文的文件名
        file_path = base_dir + line.split(" ..")[1]
        # 根据文件路径，读取文件的内容
        with open(file_path.rstrip('\n'), mode='r', encoding='gbk', errors="ignore") as f:
            content = f.read()
        # 数据清洗
        res = data_clean(content)
        # 分词
        result = word_cut(res)
        # 保存邮件的标签、邮件内容处理之后的结果
        contents.append(result)
        labels.append(label)
        # 更新进度
        tq.update()
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, random_state=1)
    # 将处理结果保存为CSV文件
    pd.DataFrame({"content": x_train, "label": y_train}).to_csv("./data/train_set.csv")
    pd.DataFrame({"content": x_test, "label": y_test}).to_csv("./data/test_set.csv")

if __name__ == '__main__':
    data_get()