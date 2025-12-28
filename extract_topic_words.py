import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LDA主题词提取")
print("="*80)

# ===========================
# 1. 加载数据和词典
# ===========================
print("\n[步骤1] 加载数据和词典")
print("-"*80)

# 读取词袋向量数据
df = pd.read_csv('cn_bow.csv')

# 读取词典（词ID到词的映射）
print("正在加载词典...")
word_dict = {}
try:
    with open('cn_dictionary.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: 词 词ID（空格分隔）
            parts = line.rsplit(' ', 1)  # 从右边分割一次，防止词中包含空格
            if len(parts) == 2:
                word = parts[0]
                word_id = int(parts[1])
                word_dict[word_id] = word
    print(f"✓ 词典加载完成，共 {len(word_dict):,} 个词")
except FileNotFoundError:
    print("⚠️  未找到词典文件 cn_dictionary.txt")
    print("   将使用词ID代替实际词汇")
    word_dict = None
except Exception as e:
    print(f"⚠️  加载词典时出错: {e}")
    print("   将使用词ID代替实际词汇")
    word_dict = None

# ===========================
# 2. 构建文档-词矩阵
# ===========================
print("\n[步骤2] 构建文档-词矩阵")
print("-"*80)

def parse_bow_vector(bow_string):
    """将 '词ID:频次 词ID:频次' 格式转换为字典"""
    bow_dict = {}
    for item in bow_string.split():
        word_id, count = item.split(':')
        bow_dict[int(word_id)] = int(count)
    return bow_dict

# 解析词袋向量
all_word_ids = set()
bow_vectors = []
for bow_str in df['bow_vector']:
    bow_dict = parse_bow_vector(bow_str)
    bow_vectors.append(bow_dict)
    all_word_ids.update(bow_dict.keys())

# 创建词ID到列索引的映射
word_id_to_idx = {word_id: idx for idx, word_id in enumerate(sorted(all_word_ids))}
idx_to_word_id = {idx: word_id for word_id, idx in word_id_to_idx.items()}
n_features = len(all_word_ids)

print(f"文档数量: {len(bow_vectors):,}")
print(f"词汇表大小: {n_features:,}")

# 构建稀疏矩阵
print("正在构建稀疏矩阵...")
X = lil_matrix((len(bow_vectors), n_features), dtype=np.float32)
for i, bow_dict in enumerate(bow_vectors):
    for word_id, count in bow_dict.items():
        X[i, word_id_to_idx[word_id]] = count

X = X.tocsr()
print(f"✓ 矩阵构建完成: {X.shape}")
print(f"  稀疏度: {1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")

# ===========================
# 3. 训练LDA模型
# ===========================
print("\n[步骤3] 训练LDA模型（最优配置：4个主题）")
print("-"*80)

n_topics = 4  # 根据实验一的结果，最优主题数为4
n_top_words = 20  # 每个主题提取的关键词数量

print(f"训练参数：")
print(f"  - 主题数量: {n_topics}")
print(f"  - 每个主题提取: 前{n_top_words}个关键词")

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=50,  # 增加迭代次数以提高质量
    learning_method='online',
    learning_decay=0.7,
    learning_offset=50.0,
    batch_size=512,
    random_state=42,
    n_jobs=1,
    verbose=0
)

print("正在训练LDA模型...")
lda.fit(X)
perplexity = lda.perplexity(X)

print(f"✓ 训练完成！")
print(f"  困惑度: {perplexity:.2f}")

# ===========================
# 4. 提取主题词
# ===========================
print("\n[步骤4] 提取各主题的关键词")
print("="*80)

def get_top_words(model, feature_names, n_top_words=20):
    """提取每个主题的top词"""
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = []
        for idx in top_indices:
            word_id = idx_to_word_id[idx]
            word = word_dict.get(word_id, f"词ID_{word_id}") if word_dict else f"词ID_{word_id}"
            weight = topic[idx]
            top_words.append((word, weight, word_id))
        topics.append(top_words)
    return topics

# 提取主题词
topics = get_top_words(lda, word_id_to_idx, n_top_words)

# ===========================
# 5. 展示和保存结果
# ===========================
print("\n" + "="*80)
print("主题词提取结果")
print("="*80)

# 准备保存的数据
all_topics_data = []

for topic_idx, topic_words in enumerate(topics):
    print(f"\n【主题 {topic_idx + 1}】")
    print("-"*80)

    # 显示前20个词
    print(f"{'排名':<6} {'词汇':<20} {'权重':<12} {'词ID':<10}")
    print("-"*80)

    for rank, (word, weight, word_id) in enumerate(topic_words, 1):
        print(f"{rank:<6} {word:<20} {weight:<12.6f} {word_id:<10}")

        # 保存数据
        all_topics_data.append({
            '主题编号': topic_idx + 1,
            '排名': rank,
            '词汇': word,
            '权重': weight,
            '词ID': word_id
        })

    # 显示前10个词的简要列表
    top10_words = [word for word, _, _ in topic_words[:10]]
    print(f"\n  🔑 Top 10关键词: {', '.join(top10_words)}")

# 保存到CSV
topics_df = pd.DataFrame(all_topics_data)
topics_df.to_csv('lda_topic_words.csv', index=False, encoding='utf-8-sig')
print("\n" + "="*80)
print("✓ 主题词已保存到: lda_topic_words.csv")

# ===========================
# 6. 文档-主题分布分析
# ===========================
print("\n[步骤5] 分析文档-主题分布")
print("-"*80)

# 获取文档的主题分布
doc_topic_dist = lda.transform(X)

# 为每个文档分配主要主题
dominant_topics = np.argmax(doc_topic_dist, axis=1)

# 统计每个主题的文档数量
topic_counts = np.bincount(dominant_topics, minlength=n_topics)

print("\n各主题的文档分布:")
print("-"*80)
for topic_idx in range(n_topics):
    count = topic_counts[topic_idx]
    percentage = count / len(dominant_topics) * 100
    print(f"主题 {topic_idx + 1}: {count:>5} 篇文档 ({percentage:>5.1f}%)")

# 为原始数据添加主题标签
df['主要主题'] = dominant_topics + 1  # 从1开始编号
df['主题1_概率'] = doc_topic_dist[:, 0]
df['主题2_概率'] = doc_topic_dist[:, 1]
df['主题3_概率'] = doc_topic_dist[:, 2]
df['主题4_概率'] = doc_topic_dist[:, 3]

# 保存带主题标签的数据
df.to_csv('documents_with_topics.csv', index=False, encoding='utf-8-sig')
print("\n✓ 带主题标签的文档已保存到: documents_with_topics.csv")

# ===========================
# 7. 提取每个主题的代表性文档
# ===========================
print("\n[步骤6] 提取各主题的代表性文档")
print("-"*80)

representative_docs = []

for topic_idx in range(n_topics):
    print(f"\n【主题 {topic_idx + 1} 的代表性文档】")
    print("-"*80)

    # 找出该主题概率最高的3个文档
    topic_probs = doc_topic_dist[:, topic_idx]
    top_doc_indices = topic_probs.argsort()[-3:][::-1]

    for rank, doc_idx in enumerate(top_doc_indices, 1):
        doc_date = df.iloc[doc_idx]['date']
        doc_label = df.iloc[doc_idx]['label']
        doc_prob = topic_probs[doc_idx]

        print(f"\n  文档 {rank}:")
        print(f"    日期: {doc_date}")
        print(f"    标签: {doc_label}")
        print(f"    主题概率: {doc_prob:.4f}")

        representative_docs.append({
            '主题编号': topic_idx + 1,
            '排名': rank,
            '文档索引': doc_idx,
            '日期': doc_date,
            '标签': doc_label,
            '主题概率': doc_prob
        })

# 保存代表性文档
rep_docs_df = pd.DataFrame(representative_docs)
rep_docs_df.to_csv('representative_documents.csv', index=False, encoding='utf-8-sig')
print("\n" + "="*80)
print("✓ 代表性文档已保存到: representative_documents.csv")

# ===========================
# 8. 生成文本报告
# ===========================
print("\n[步骤7] 生成主题分析报告")
print("-"*80)

report = f"""
LDA主题模型分析报告
{"="*80}

1. 模型配置
   - 主题数量: {n_topics}
   - 文档数量: {len(bow_vectors):,}
   - 词汇表大小: {n_features:,}
   - 困惑度: {perplexity:.2f}

2. 各主题关键词（Top 10）
{"="*80}

"""

for topic_idx, topic_words in enumerate(topics):
    report += f"\n主题 {topic_idx + 1}:\n"
    top10_words = [word for word, _, _ in topic_words[:10]]
    report += f"  关键词: {', '.join(top10_words)}\n"
    report += f"  文档数: {topic_counts[topic_idx]} 篇 ({topic_counts[topic_idx]/len(dominant_topics)*100:.1f}%)\n"

report += f"""
{"="*80}

3. 文档分布统计
   - 主题1: {topic_counts[0]:>5} 篇 ({topic_counts[0]/len(dominant_topics)*100:>5.1f}%)
   - 主题2: {topic_counts[1]:>5} 篇 ({topic_counts[1]/len(dominant_topics)*100:>5.1f}%)
   - 主题3: {topic_counts[2]:>5} 篇 ({topic_counts[2]/len(dominant_topics)*100:>5.1f}%)
   - 主题4: {topic_counts[3]:>5} 篇 ({topic_counts[3]/len(dominant_topics)*100:>5.1f}%)

4. 生成文件
   ✓ lda_topic_words.csv - 所有主题的关键词列表
   ✓ documents_with_topics.csv - 带主题标签的完整文档
   ✓ representative_documents.csv - 各主题的代表性文档
   ✓ lda_topic_analysis_report.txt - 本分析报告

{"="*80}
"""

with open('lda_topic_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✓ 分析报告已保存到: lda_topic_analysis_report.txt")

# ===========================
# 9. 主题-主题相似度分析
# ===========================
print("\n[步骤8] 主题间相似度分析")
print("-"*80)

from sklearn.metrics.pairwise import cosine_similarity

# 计算主题间的余弦相似度
topic_vectors = lda.components_
similarity_matrix = cosine_similarity(topic_vectors)

print("\n主题间余弦相似度矩阵:")
print("-"*80)
print(f"{'':>10}", end='')
for i in range(n_topics):
    print(f"主题{i+1:>6}", end='')
print()
print("-"*80)

for i in range(n_topics):
    print(f"主题{i+1:>6}   ", end='')
    for j in range(n_topics):
        print(f"{similarity_matrix[i][j]:>8.4f}", end='')
    print()

# 找出最相似和最不相似的主题对
similarities = []
for i in range(n_topics):
    for j in range(i+1, n_topics):
        similarities.append((i+1, j+1, similarity_matrix[i][j]))

similarities.sort(key=lambda x: x[2], reverse=True)

print("\n主题对相似度排序:")
print("-"*80)
for topic1, topic2, sim in similarities:
    print(f"主题{topic1} ↔ 主题{topic2}: {sim:.4f}")

# ===========================
# 10. 总结
# ===========================
print("\n" + "="*80)
print("✅ 主题词提取完成！")
print("="*80)

print("\n📁 生成的文件:")
print("  1. lda_topic_words.csv - 各主题的Top 20关键词")
print("  2. documents_with_topics.csv - 带主题标签的完整数据")
print("  3. representative_documents.csv - 各主题的代表性文档")
print("  4. lda_topic_analysis_report.txt - 主题分析报告")

print("\n📊 主题分布:")
for topic_idx in range(n_topics):
    count = topic_counts[topic_idx]
    percentage = count / len(dominant_topics) * 100
    bar_length = int(percentage / 2)
    bar = '█' * bar_length
    print(f"  主题{topic_idx+1}: {bar} {count}篇 ({percentage:.1f}%)")

print("\n💡 使用建议:")
print("  - 查看 lda_topic_words.csv 了解各主题的关键词")
print("  - 查看 representative_documents.csv 了解各主题的典型文档")
print("  - 使用 documents_with_topics.csv 进行进一步分析")
print("  - 根据主题关键词为各主题命名")

print("\n" + "="*80)
