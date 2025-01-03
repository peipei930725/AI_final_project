import os
import json
import jieba
from collections import defaultdict
from math import log

# 嘗試啟用 Paddle 模式，若無法啟用則回退至預設模式
try:
    jieba.enable_paddle()  # 啟動 Paddle 模式
    PADDLE_MODE = True  # 如果 Paddle 模式成功啟用，設置為 True
except Exception as e:
    print("Paddle 模式無法啟用，將回退至預設模式。")
    PADDLE_MODE = False  # 如果 Paddle 模式無法啟用，設置為 False

# === 步驟 1: 資料載入 ===
def load_json(file_path):
    """從 JSON 檔案中載入文字資料。"""
    # 開啟指定路徑的 JSON 檔案並載入為字典
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_text(file_path):
    """從純文字檔案中載入資料。"""
    # 開啟純文字檔案並逐行讀取，去除空行
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def load_all_data_from_dir(directory):
    """從目錄中的所有 JSON 檔案中載入並保留股票代碼。"""
    combined_data = {}  # 初始化空字典以存儲合併資料
    for filename in os.listdir(directory):  # 遍歷目錄中的所有檔案
        if filename.endswith('.json'):  # 只處理以 .json 結尾的檔案
            stock_code = filename.split('.')[0]  # 提取股票代碼作為鍵值
            file_path = os.path.join(directory, filename)  # 獲取檔案完整路徑
            data = load_json(file_path)  # 載入 JSON 檔案
            combined_data[stock_code] = data  # 將股票代碼與其資料對應
    return combined_data

# === 步驟 2: 資料預處理 ===
def preprocess(text):
    """對文字進行預處理：分詞並移除停用詞和介系詞。"""
    stopwords_path = os.path.join("extra_dict", "stopwords.txt")  # 停用詞檔案路徑
    stopwords = set(load_text(stopwords_path))  # 載入停用詞並轉換為集合
    if PADDLE_MODE:
        tokens = jieba.cut(text, use_paddle=True)  # 使用 Paddle 模式分詞
    else:
        tokens = jieba.cut(text)  # 使用預設模式分詞
    # 過濾停用詞、介系詞，並去除空字串或空白分詞
    return [token for token in tokens if token.strip() and token not in stopwords]

# === 步驟 3: 倒排索引構建 ===
def build_inverted_index(data):
    """從多股票 JSON 資料構建倒排索引。"""
    index = defaultdict(list)  # 倒排索引初始化為預設列表字典
    documents = []  # 用於存儲所有文件資訊
    seen_docs = set()  # 用於追蹤已處理的文件內容

    for stock_code, stock_data in data.items():  # 遍歷每支股票的資料
        for date, news_list in stock_data.items():  # 遍歷該股票資料中的每個日期和新聞列表
            for doc_id, content in enumerate(news_list):  # 遍歷新聞列表中的每篇文章
                full_doc_id = f"{stock_code}-{date}-{doc_id}"  # 構造唯一文件 ID
                if content in seen_docs:  # 檢查是否已處理過相同內容
                    continue  # 如果已處理過，跳過
                seen_docs.add(content)  # 添加到已處理集合
                documents.append((full_doc_id, content))  # 將文件 ID 和內容加入文件列表
                tokens = preprocess(content)  # 對文章內容進行分詞和預處理
                for token in set(tokens):  # 遍歷文章中的唯一詞語
                    index[token].append(full_doc_id)  # 將文件 ID 添加到詞語的索引列表中

    return index, documents  # 返回倒排索引和文件列表

# === 步驟 4: 搜尋實作 ===
def search(query, index, documents):
    """搜尋與查詢匹配的文件，使用 TF-IDF 計算分數。"""
    query_tokens = preprocess(query)  # 對查詢語句進行分詞和預處理
    doc_scores = defaultdict(float)  # 初始化文件得分字典（浮點數）

    for token in query_tokens:  # 遍歷查詢中的每個詞語
        if token in index:  # 如果詞語存在於倒排索引中
            idf = log(len(documents) / len(index[token]))  # 計算逆文件頻率（IDF），加 1 防止分母為 0
            for full_doc_id in index[token]:  # 遍歷包含該詞語的所有文件
                # 提取文檔內容
                doc_content = next(doc[1] for doc in documents if doc[0] == full_doc_id)
                # 計算詞頻（TF）：詞語在文章中的出現次數除以文章的總詞數
                token_count = doc_content.count(token)  # 詞語出現的次數
                total_tokens = len(preprocess(doc_content))  # 文檔的總詞數
                tf = token_count / total_tokens  # 詞頻
                doc_scores[full_doc_id] += tf * idf  # 使用 TF-IDF 更新得分

    # 按文件得分降序排序
    sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(full_doc_id, doc_scores[full_doc_id]) for full_doc_id, _ in sorted_results]

# === 步驟 5: 結果顯示 ===
def display_results(results, documents, query_tokens):
    """顯示搜尋結果和文件細節，並將關鍵字標註為紅色。"""
    doc_map = {doc[0]: doc[1] for doc in documents}  # 建立文件 ID 到內容的映射
    for idx, (full_doc_id, score) in enumerate(results[:20], start=1):  # 遍歷搜尋結果，只顯示前 20 個
        stock_code, date, doc_id = full_doc_id.split('-')  # 拆分文件 ID 獲取股票代碼、日期和序號
        content = doc_map[full_doc_id].strip()
        for token in query_tokens:
            content = content.replace(token, f"\033[91m{token}\033[0m")  # 將關鍵字標註為紅色
        print(f"{idx}. [股票代碼: {stock_code}] [日期: {date}] {content} (分數: {score:.2f})")  # 輸出結果

# === 步驟 6: 主程式 ===
if __name__ == "__main__":
    # 從資料夾中載入數據集
    data_directory = "news"  # 資料目錄路徑
    if not os.path.exists(data_directory):  # 檢查資料目錄是否存在
        print(f"資料目錄未找到: {data_directory}")
        exit(1)

    print("正在從資料目錄中的所有 JSON 檔案載入數據...")
    data = load_all_data_from_dir(data_directory)  # 載入資料目錄中的所有 JSON 檔案

    print("正在構建倒排索引...")
    inverted_index, documents = build_inverted_index(data)  # 構建倒排索引

    print("搜尋引擎已準備就緒！\n")

    # 互動式搜尋
    while True:
        query = input("輸入搜尋查詢（或輸入 'exit' 離開）：")  # 提示使用者輸入查詢
        if query.lower() == 'exit':  # 如果輸入 'exit'，退出迴圈
            break
        results = search(query, inverted_index, documents)  # 執行查詢
        if results:  # 如果有結果
            print("\n搜尋結果：")
            display_results(results, documents, preprocess(query))  # 顯示結果並標註關鍵字
        else:
            print("\n未找到相關結果。")  # 如果沒有結果，提示使用者