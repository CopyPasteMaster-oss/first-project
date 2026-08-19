# -*- coding: utf-8 -*-
"""
垃圾邮件分类器 - Flask Web 服务
================================
把训练好的朴素贝叶斯模型封装成 Web 接口：
  GET  /              -> 返回网页表单（email_submit.html）
  GET  /email_submit  -> 返回网页表单（与课程对应）
  POST /email_handle  -> 接收邮件内容，返回分类结果（JSON）

运行方式（在 web 目录下）：
  D:/jiqixuexi/.venv/Scripts/python.exe app.py
然后浏览器打开 http://127.0.0.1:5000
"""
import sys
from pathlib import Path

import joblib
from flask import Flask, request, jsonify, send_file
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------
# 路径设置（不管从哪个目录运行都能找到模型和工具函数）
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../ML_project1/web
PROJECT_ROOT = BASE_DIR.parent.parent               # .../jiqixuexi
MODEL_DIR = BASE_DIR.parent / "train_model" / "data"  # 模型存放目录

# 让 `from ML_project1.utils.data_pro import ...` 生效
sys.path.insert(0, str(PROJECT_ROOT))

from ML_project1.utils.data_pro import data_clean, word_cut

# ---------------------------------------------------------------
# 加载模型（只加载一次，避免每次请求都重新读文件）
# ---------------------------------------------------------------
word_list = joblib.load(str(MODEL_DIR / "word_list.pth"))   # 训练时的词典
vector = CountVectorizer(vocabulary=word_list)              # 用同一个词典
model = joblib.load(str(MODEL_DIR / "best_model.pth"))      # 训练好的模型

app = Flask(__name__)


def predict(email_text: str) -> str:
    """把一段邮件文字变成分类结果：'垃圾邮件' / '正常邮件'"""
    # ① 清洗：只留中文 + 繁体转简体
    content = data_clean(email_text)
    # ② 分词：jieba 按词性过滤
    content = [word_cut(content)]
    # ③ 向量化：文字 -> 数字（必须用训练时的词典！）
    x = vector.transform(content).toarray()
    # ④ 预测：1 表示垃圾邮件，0 表示正常邮件
    y = model.predict(x)[0]
    return "垃圾邮件" if y == 1 else "正常邮件"


@app.route("/")
def index():
    """首页：直接展示提交表单网页"""
    return send_file(BASE_DIR / "email_submit.html")


@app.route("/email_submit", methods=["GET"])
def email_submit():
    """（与课程 04 对应）返回提交表单网页"""
    return send_file(BASE_DIR / "email_submit.html")


@app.route("/email_handle", methods=["POST"])
def email_handle():
    """接收邮件内容，返回预测结果"""
    # 兼容两种提交方式：表单 (form) 和 JSON
    if request.is_json:
        content = request.json.get("content", "")
    else:
        content = request.form.get("content", "")

    if not content or not content.strip():
        return jsonify({"result": "请输入邮件内容"}), 400

    result = predict(content)
    return jsonify({"result": result})


if __name__ == "__main__":
    # debug=True 改代码自动重启，学习阶段很方便
    app.run(host="0.0.0.0", port=5000, debug=True)
