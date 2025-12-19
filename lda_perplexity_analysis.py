import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import lil_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("正在读取数据...")
df = pd.read_csv('cn_bow.csv')

# 解析bow_vector列，将稀疏向量转换为稀疏矩阵
print("正在解析词袋向量...")
def parse_bow_vector(bow_string):
    """将 '词ID:频次 词ID:频次' 格式转换为字典"""
    bow_dict = {}
    for item in bow_string.split():
        word_id, count = item.split(':')
        bow_dict[int(word_id)] = int(count)
    return bow_dict

# 获取所有词ID
all_word_ids = set()
bow_vectors = []
for bow_str in df['bow_vector']:
    bow_dict = parse_bow_vector(bow_str)
    bow_vectors.append(bow_dict)
    all_word_ids.update(bow_dict.keys())

# 创建词ID到列索引的映射
word_id_to_idx = {word_id: idx for idx, word_id in enumerate(sorted(all_word_ids))}
n_features = len(all_word_ids)

print(f"文档数量: {len(bow_vectors)}")
print(f"词汇表大小: {n_features}")

# 使用稀疏矩阵（lil_matrix）来节省内存
print("正在构建稀疏矩阵...")
X = lil_matrix((len(bow_vectors), n_features))
for i, bow_dict in enumerate(bow_vectors):
    for word_id, count in bow_dict.items():
        X[i, word_id_to_idx[word_id]] = count

# 转换为csr格式以提高计算效率
X = X.tocsr()
print(f"文档-词矩阵形状: {X.shape}")
print(f"矩阵稀疏度: {1 - X.nnz / (X.shape[0] * X.shape[1]):.4%}")

# 设置要测试的主题数量范围
topic_range = range(2, 21, 1)  # 从2到20个主题
perplexities = []
exp_neg_log_perplexities = []

print("\n开始计算不同主题数量下的困惑度...")
for n_topics in topic_range:
    print(f"正在计算 {n_topics} 个主题的LDA模型...")

    # 创建并训练LDA模型（使用n_jobs=1避免内存问题）
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,  # 减少迭代次数以加快速度
        learning_method='online',  # 使用在线学习以节省内存
        random_state=42,
        n_jobs=1,  # 禁用并行处理避免pickle和内存问题
        batch_size=128,
        evaluate_every=5,
        verbose=0
    )

    lda.fit(X)

    # 计算困惑度
    perplexity = lda.perplexity(X)
    perplexities.append(perplexity)

    # 计算 e^(-log(perplexity)) = 1/perplexity
    # 或者理解为 e^(-perplexity_score)，这里使用 e^(-log(perplexity))
    exp_neg_log_perplexity = np.exp(-np.log(perplexity))
    exp_neg_log_perplexities.append(exp_neg_log_perplexity)

    print(f"  困惑度: {perplexity:.4f}, e^(-log(perplexity)): {exp_neg_log_perplexity:.6f}")

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1: 困惑度曲线
axes[0].plot(list(topic_range), perplexities, 'bo-', linewidth=2, markersize=6)
axes[0].set_xlabel('主题个数', fontsize=12)
axes[0].set_ylabel('困惑度 (Perplexity)', fontsize=12)
axes[0].set_title('LDA模型困惑度与主题个数的关系', fontsize=14)
axes[0].grid(True, alpha=0.3)

# 子图2: e的负对数次方曲线
axes[1].plot(list(topic_range), exp_neg_log_perplexities, 'ro-', linewidth=2, markersize=6)
axes[1].set_xlabel('主题个数', fontsize=12)
axes[1].set_ylabel('e^(-log(Perplexity))', fontsize=12)
axes[1].set_title('e的负对数困惑度与主题个数的关系', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lda_perplexity_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存为 'lda_perplexity_analysis.png'")

# 输出结果表格
results_df = pd.DataFrame({
    '主题个数': list(topic_range),
    '困惑度': perplexities,
    'e^(-log(Perplexity))': exp_neg_log_perplexities
})

print("\n" + "="*60)
print("详细结果:")
print("="*60)
print(results_df.to_string(index=False))

# 保存结果到CSV
results_df.to_csv('lda_perplexity_results.csv', index=False, encoding='utf-8-sig')
print(f"\n结果已保存为 'lda_perplexity_results.csv'")

# 找出最优主题数（困惑度最低）
optimal_idx = np.argmin(perplexities)
optimal_topics = list(topic_range)[optimal_idx]
print(f"\n最优主题个数（困惑度最低）: {optimal_topics}")
print(f"对应困惑度: {perplexities[optimal_idx]:.4f}")

plt.show()
