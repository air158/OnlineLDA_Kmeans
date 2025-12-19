import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import lil_matrix
import warnings
import time
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("在线LDA (Online LDA) 困惑度分析")
print("="*70)

# 读取数据
print("\n[1/5] 正在读取数据...")
start_time = time.time()
df = pd.read_csv('cn_bow.csv')
print(f"✓ 数据读取完成 ({time.time()-start_time:.2f}秒)")

# 解析bow_vector列
print("\n[2/5] 正在解析词袋向量...")
start_time = time.time()

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

print(f"✓ 解析完成 ({time.time()-start_time:.2f}秒)")
print(f"  - 文档数量: {len(bow_vectors):,}")
print(f"  - 词汇表大小: {n_features:,}")

# 构建稀疏矩阵
print("\n[3/5] 正在构建稀疏矩阵...")
start_time = time.time()
X = lil_matrix((len(bow_vectors), n_features), dtype=np.float32)
for i, bow_dict in enumerate(bow_vectors):
    for word_id, count in bow_dict.items():
        X[i, word_id_to_idx[word_id]] = count

# 转换为csr格式以提高计算效率
X = X.tocsr()
sparsity = 1 - X.nnz / (X.shape[0] * X.shape[1])
print(f"✓ 矩阵构建完成 ({time.time()-start_time:.2f}秒)")
print(f"  - 矩阵形状: {X.shape}")
print(f"  - 稀疏度: {sparsity:.2%}")
print(f"  - 非零元素: {X.nnz:,}")

# 设置要测试的主题数量范围
topic_range = list(range(2, 21, 2))  # 2, 4, 6, 8, ..., 20
perplexities = []
exp_neg_log_perplexities = []

print(f"\n[4/5] 开始在线LDA训练（共{len(topic_range)}个主题数）...")
print("="*70)

total_start = time.time()
for idx, n_topics in enumerate(topic_range, 1):
    iter_start = time.time()
    print(f"\n[{idx}/{len(topic_range)}] 主题数 = {n_topics}")

    # 使用在线LDA (Online LDA)
    olda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,                    # 减少迭代次数
        learning_method='online',       # 在线学习方法
        learning_decay=0.7,             # 学习率衰减
        learning_offset=50.0,           # 学习率偏移
        batch_size=512,                 # 批次大小
        random_state=42,
        n_jobs=1,                       # 单线程避免内存问题
        verbose=0
    )

    # 训练模型
    print("  - 训练中...", end=" ", flush=True)
    olda.fit(X)

    # 计算困惑度
    perplexity = olda.perplexity(X)
    perplexities.append(perplexity)

    # 计算 e^(-log(perplexity)) = 1/perplexity
    exp_neg_log_perplexity = np.exp(-np.log(perplexity))
    exp_neg_log_perplexities.append(exp_neg_log_perplexity)

    elapsed = time.time() - iter_start
    print(f"完成 ({elapsed:.1f}秒)")
    print(f"  - 困惑度: {perplexity:.2f}")
    print(f"  - e^(-log(Perplexity)): {exp_neg_log_perplexity:.6f}")

total_time = time.time() - total_start
print("\n" + "="*70)
print(f"✓ 所有模型训练完成！总耗时: {total_time:.1f}秒 (平均 {total_time/len(topic_range):.1f}秒/模型)")

# 创建可视化图表
print("\n[5/5] 正在生成可视化图表...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 子图1: 困惑度曲线
axes[0].plot(topic_range, perplexities, 'bo-', linewidth=2.5, markersize=8,
             markerfacecolor='lightblue', markeredgewidth=2)
axes[0].set_xlabel('主题个数', fontsize=13, fontweight='bold')
axes[0].set_ylabel('困惑度 (Perplexity)', fontsize=13, fontweight='bold')
axes[0].set_title('在线LDA模型困惑度与主题个数的关系', fontsize=14, fontweight='bold', pad=15)
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=1)
axes[0].set_xticks(topic_range)
axes[0].tick_params(labelsize=11)

# 标注最小值
min_idx = np.argmin(perplexities)
axes[0].scatter([topic_range[min_idx]], [perplexities[min_idx]],
                color='red', s=200, zorder=5, marker='*',
                edgecolors='darkred', linewidths=2)
axes[0].annotate(f'最小值\n主题数={topic_range[min_idx]}\n困惑度={perplexities[min_idx]:.2f}',
                xy=(topic_range[min_idx], perplexities[min_idx]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'),
                fontsize=10, fontweight='bold')

# 子图2: e的负对数次方曲线
axes[1].plot(topic_range, exp_neg_log_perplexities, 'ro-', linewidth=2.5,
             markersize=8, markerfacecolor='lightcoral', markeredgewidth=2)
axes[1].set_xlabel('主题个数', fontsize=13, fontweight='bold')
axes[1].set_ylabel('e^(-log(Perplexity))', fontsize=13, fontweight='bold')
axes[1].set_title('e的负对数困惑度与主题个数的关系', fontsize=14, fontweight='bold', pad=15)
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=1)
axes[1].set_xticks(topic_range)
axes[1].tick_params(labelsize=11)

# 标注最大值
max_idx = np.argmax(exp_neg_log_perplexities)
axes[1].scatter([topic_range[max_idx]], [exp_neg_log_perplexities[max_idx]],
                color='darkgreen', s=200, zorder=5, marker='*',
                edgecolors='green', linewidths=2)
axes[1].annotate(f'最大值\n主题数={topic_range[max_idx]}\n值={exp_neg_log_perplexities[max_idx]:.6f}',
                xy=(topic_range[max_idx], exp_neg_log_perplexities[max_idx]),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green'),
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('olda_perplexity_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 图表已保存为 'olda_perplexity_analysis.png'")

# 保存结果
results_df = pd.DataFrame({
    '主题个数': topic_range,
    '困惑度': perplexities,
    'e^(-log(Perplexity))': exp_neg_log_perplexities
})

print("\n" + "="*70)
print("详细结果:")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

results_df.to_csv('olda_perplexity_results.csv', index=False, encoding='utf-8-sig')
print("\n✓ 结果已保存为 'olda_perplexity_results.csv'")

# 分析最优主题数
optimal_idx = np.argmin(perplexities)
optimal_topics = topic_range[optimal_idx]

print("\n" + "="*70)
print("最优主题数分析:")
print("="*70)
print(f"📊 最优主题个数（困惑度最低）: {optimal_topics}")
print(f"📉 对应困惑度: {perplexities[optimal_idx]:.2f}")
print(f"📈 对应 e^(-log(Perplexity)): {exp_neg_log_perplexities[optimal_idx]:.6f}")
print("="*70)

print("\n✅ 分析完成！")
plt.show()
