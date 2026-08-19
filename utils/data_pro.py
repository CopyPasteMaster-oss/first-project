# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   ML_project
# FileName:      data_pro.py
# ---------------------------------------------------------------------------
import re
import zhconv
import jieba.posseg

def data_clean(content):
    # 将邮件中非中文替换成空字符串
    res = re.sub(r'[^\u4e00-\u9fa5]', '', content)
    # 把中文繁体替换中简体
    res = zhconv.convert(res, locale="zh-CN")
    return res


def word_cut(content):
    # 对数据按照词性分词 ：['n', 'nr', 'ns', 'nt', 'v', 'a']
    result = jieba.posseg.lcut(content)
    # 创建一个列表,保存分词之后的结果
    word_list = []
    # 按照词性分词
    for word, psg in result:
        if psg in ['n', 'nr', 'ns', 'nt', 'v', 'a']:
            word_list.append(word)
    return " ".join(word_list)