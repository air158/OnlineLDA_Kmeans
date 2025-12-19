import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
import warnings
import time
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("文本聚类算法对比分析：KMeans vs Single-pass")
print("="*80)

# ===========================
# 1. 数据加载和预处理
# ===========================
print("\n[步骤1] 数据加载和预处理")
print("-"*80)
start_time = time.time()

df = pd.read_csv('cn_bow.csv')
print(f"✓ 数据读取完成")

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

# 创建词ID映射
word_id_to_idx = {word_id: idx for idx, word_id in enumerate(sorted(all_word_ids))}
n_features = len(all_word_ids)

print(f"  - 文档数量: {len(bow_vectors):,}")
print(f"  - 词汇表大小: {n_features:,}")

# 构建稀疏矩阵
X = lil_matrix((len(bow_vectors), n_features), dtype=np.float32)
for i, bow_dict in enumerate(bow_vectors):
    for word_id, count in bow_dict.items():
        X[i, word_id_to_idx[word_id]] = count

X = X.tocsr()
sparsity = 1 - X.nnz / (X.shape[0] * X.shape[1])
print(f"  - 矩阵形状: {X.shape}")
print(f"  - 稀疏度: {sparsity:.2%}")
print(f"✓ 预处理完成 (耗时: {time.time()-start_time:.2f}秒)")

# ===========================
# 2. 降维以便可视化和加速计算
# ===========================
print("\n[步骤2] 使用SVD进行降维")
print("-"*80)
start_time = time.time()

n_components = 100  # 降到100维
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X)
explained_variance = svd.explained_variance_ratio_.sum()

print(f"  - 降维后维度: {X_reduced.shape}")
print(f"  - 保留方差比例: {explained_variance:.2%}")
print(f"✓ 降维完成 (耗时: {time.time()-start_time:.2f}秒)")

# ===========================
# 3. Single-pass 聚类算法实现
# ===========================
print("\n[步骤3] Single-pass 聚类算法")
print("-"*80)

class SinglePassCluster:
    """Single-pass 聚类算法"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.cluster_centers_ = []
        self.labels_ = []
        self.n_clusters = 0

    def fit(self, X):
        """训练Single-pass聚类模型"""
        start_time = time.time()
        self.cluster_centers_ = []
        self.labels_ = np.zeros(X.shape[0], dtype=int)

        for i, doc in enumerate(X):
            if i == 0:
                # 第一个文档作为第一个聚类中心
                self.cluster_centers_.append(doc.copy())
                self.labels_[i] = 0
                self.n_clusters = 1
            else:
                # 计算与所有聚类中心的相似度
                max_similarity = -1
                best_cluster = -1

                for j, center in enumerate(self.cluster_centers_):
                    similarity = self._cosine_similarity(doc, center)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_cluster = j

                # 如果相似度超过阈值，加入该聚类
                if max_similarity >= self.threshold:
                    self.labels_[i] = best_cluster
                    # 更新聚类中心
                    cluster_size = np.sum(self.labels_[:i+1] == best_cluster)
                    self.cluster_centers_[best_cluster] = (
                        (self.cluster_centers_[best_cluster] * (cluster_size - 1) + doc) / cluster_size
                    )
                else:
                    # 创建新聚类
                    self.cluster_centers_.append(doc.copy())
                    self.labels_[i] = self.n_clusters
                    self.n_clusters += 1

            # 每1000个文档打印进度
            if (i + 1) % 1000 == 0:
                print(f"    处理进度: {i+1}/{X.shape[0]}, 当前聚类数: {self.n_clusters}", end='\r')

        self.fit_time = time.time() - start_time
        print(f"\n  ✓ Single-pass聚类完成")
        print(f"    - 耗时: {self.fit_time:.2f}秒")
        print(f"    - 聚类数量: {self.n_clusters}")
        return self

    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)

# 测试不同阈值的Single-pass
thresholds = [0.3, 0.5, 0.7]
singlepass_results = []

for threshold in thresholds:
    print(f"\n  测试阈值: {threshold}")
    sp = SinglePassCluster(threshold=threshold)
    sp.fit(X_reduced)

    # 评估指标
    if sp.n_clusters > 1 and sp.n_clusters < len(X_reduced):
        silhouette = silhouette_score(X_reduced, sp.labels_)
        calinski = calinski_harabasz_score(X_reduced, sp.labels_)
        davies = davies_bouldin_score(X_reduced, sp.labels_)
    else:
        silhouette = -1
        calinski = 0
        davies = float('inf')

    singlepass_results.append({
        'threshold': threshold,
        'n_clusters': sp.n_clusters,
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies,
        'time': sp.fit_time,
        'labels': sp.labels_.copy()
    })

    print(f"    - 轮廓系数: {silhouette:.4f}")
    print(f"    - Calinski-Harabasz指数: {calinski:.2f}")
    print(f"    - Davies-Bouldin指数: {davies:.4f}")

# 选择最佳Single-pass结果
best_sp = max(singlepass_results, key=lambda x: x['silhouette'])
print(f"\n  ✓ 最佳Single-pass配置: 阈值={best_sp['threshold']}, 聚类数={best_sp['n_clusters']}")

# ===========================
# 4. KMeans 聚类算法
# ===========================
print("\n[步骤4] KMeans 聚类算法")
print("-"*80)

# 测试不同K值的KMeans
k_values = [3, 4, 5, 6, 8, 10, 12, 15]
kmeans_results = []

for k in k_values:
    print(f"\n  测试K={k}")
    start_time = time.time()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_reduced)
    fit_time = time.time() - start_time

    # 评估指标
    silhouette = silhouette_score(X_reduced, labels)
    calinski = calinski_harabasz_score(X_reduced, labels)
    davies = davies_bouldin_score(X_reduced, labels)
    inertia = kmeans.inertia_

    kmeans_results.append({
        'k': k,
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies,
        'inertia': inertia,
        'time': fit_time,
        'labels': labels.copy(),
        'model': kmeans
    })

    print(f"    - 耗时: {fit_time:.2f}秒")
    print(f"    - 轮廓系数: {silhouette:.4f}")
    print(f"    - Calinski-Harabasz指数: {calinski:.2f}")
    print(f"    - Davies-Bouldin指数: {davies:.4f}")
    print(f"    - 惯性: {inertia:.2f}")

# 选择最佳KMeans结果
best_kmeans = max(kmeans_results, key=lambda x: x['silhouette'])
print(f"\n  ✓ 最佳KMeans配置: K={best_kmeans['k']}")

# ===========================
# 5. 结果对比和可视化
# ===========================
print("\n[步骤5] 算法对比分析")
print("="*80)

print("\n【Single-pass 最佳结果】")
print(f"  - 阈值: {best_sp['threshold']}")
print(f"  - 聚类数量: {best_sp['n_clusters']}")
print(f"  - 轮廓系数: {best_sp['silhouette']:.4f}")
print(f"  - Calinski-Harabasz指数: {best_sp['calinski']:.2f}")
print(f"  - Davies-Bouldin指数: {best_sp['davies']:.4f}")
print(f"  - 训练时间: {best_sp['time']:.2f}秒")

print("\n【KMeans 最佳结果】")
print(f"  - K值: {best_kmeans['k']}")
print(f"  - 轮廓系数: {best_kmeans['silhouette']:.4f}")
print(f"  - Calinski-Harabasz指数: {best_kmeans['calinski']:.2f}")
print(f"  - Davies-Bouldin指数: {best_kmeans['davies']:.4f}")
print(f"  - 惯性: {best_kmeans['inertia']:.2f}")
print(f"  - 训练时间: {best_kmeans['time']:.2f}秒")

print("\n【综合对比】")
if best_kmeans['silhouette'] > best_sp['silhouette']:
    winner = "KMeans"
    print(f"  🏆 推荐算法: KMeans (K={best_kmeans['k']})")
    print(f"  📊 理由: 轮廓系数更高 ({best_kmeans['silhouette']:.4f} > {best_sp['silhouette']:.4f})")
else:
    winner = "Single-pass"
    print(f"  🏆 推荐算法: Single-pass (阈值={best_sp['threshold']})")
    print(f"  📊 理由: 轮廓系数更高 ({best_sp['silhouette']:.4f} > {best_kmeans['silhouette']:.4f})")

# ===========================
# 6. 可视化
# ===========================
print("\n[步骤6] 生成可视化图表")
print("-"*80)

# 创建综合对比图
fig = plt.figure(figsize=(18, 12))

# 1. KMeans不同K值的评估指标
ax1 = plt.subplot(3, 3, 1)
k_vals = [r['k'] for r in kmeans_results]
silhouettes = [r['silhouette'] for r in kmeans_results]
ax1.plot(k_vals, silhouettes, 'bo-', linewidth=2, markersize=8)
ax1.scatter([best_kmeans['k']], [best_kmeans['silhouette']], color='red', s=200, marker='*', zorder=5)
ax1.set_xlabel('K值', fontweight='bold')
ax1.set_ylabel('轮廓系数', fontweight='bold')
ax1.set_title('KMeans - 轮廓系数', fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
calinskis = [r['calinski'] for r in kmeans_results]
ax2.plot(k_vals, calinskis, 'go-', linewidth=2, markersize=8)
ax2.scatter([best_kmeans['k']], [best_kmeans['calinski']], color='red', s=200, marker='*', zorder=5)
ax2.set_xlabel('K值', fontweight='bold')
ax2.set_ylabel('Calinski-Harabasz指数', fontweight='bold')
ax2.set_title('KMeans - Calinski-Harabasz指数', fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 3, 3)
davies_scores = [r['davies'] for r in kmeans_results]
ax3.plot(k_vals, davies_scores, 'ro-', linewidth=2, markersize=8)
ax3.scatter([best_kmeans['k']], [best_kmeans['davies']], color='red', s=200, marker='*', zorder=5)
ax3.set_xlabel('K值', fontweight='bold')
ax3.set_ylabel('Davies-Bouldin指数', fontweight='bold')
ax3.set_title('KMeans - Davies-Bouldin指数 (越小越好)', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 2. Single-pass不同阈值的评估指标
ax4 = plt.subplot(3, 3, 4)
thresh_vals = [r['threshold'] for r in singlepass_results]
sp_silhouettes = [r['silhouette'] for r in singlepass_results]
ax4.plot(thresh_vals, sp_silhouettes, 'bo-', linewidth=2, markersize=8)
ax4.scatter([best_sp['threshold']], [best_sp['silhouette']], color='red', s=200, marker='*', zorder=5)
ax4.set_xlabel('阈值', fontweight='bold')
ax4.set_ylabel('轮廓系数', fontweight='bold')
ax4.set_title('Single-pass - 轮廓系数', fontweight='bold')
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(3, 3, 5)
sp_calinskis = [r['calinski'] for r in singlepass_results]
ax5.plot(thresh_vals, sp_calinskis, 'go-', linewidth=2, markersize=8)
ax5.scatter([best_sp['threshold']], [best_sp['calinski']], color='red', s=200, marker='*', zorder=5)
ax5.set_xlabel('阈值', fontweight='bold')
ax5.set_ylabel('Calinski-Harabasz指数', fontweight='bold')
ax5.set_title('Single-pass - Calinski-Harabasz指数', fontweight='bold')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 3, 6)
sp_nclusters = [r['n_clusters'] for r in singlepass_results]
ax6.plot(thresh_vals, sp_nclusters, 'mo-', linewidth=2, markersize=8)
ax6.scatter([best_sp['threshold']], [best_sp['n_clusters']], color='red', s=200, marker='*', zorder=5)
ax6.set_xlabel('阈值', fontweight='bold')
ax6.set_ylabel('聚类数量', fontweight='bold')
ax6.set_title('Single-pass - 聚类数量', fontweight='bold')
ax6.grid(True, alpha=0.3)

# 3. 算法对比
ax7 = plt.subplot(3, 3, 7)
algorithms = ['KMeans\n(最佳)', 'Single-pass\n(最佳)']
silhouette_comparison = [best_kmeans['silhouette'], best_sp['silhouette']]
colors = ['#2ecc71' if winner == 'KMeans' else '#3498db', '#2ecc71' if winner == 'Single-pass' else '#3498db']
bars = ax7.bar(algorithms, silhouette_comparison, color=colors, edgecolor='black', linewidth=2)
ax7.set_ylabel('轮廓系数', fontweight='bold')
ax7.set_title('算法对比 - 轮廓系数', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, silhouette_comparison):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{val:.4f}',
             ha='center', va='bottom', fontweight='bold')

ax8 = plt.subplot(3, 3, 8)
time_comparison = [best_kmeans['time'], best_sp['time']]
bars = ax8.bar(algorithms, time_comparison, color=['#e74c3c', '#f39c12'], edgecolor='black', linewidth=2)
ax8.set_ylabel('训练时间 (秒)', fontweight='bold')
ax8.set_title('算法对比 - 训练时间', fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, time_comparison):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}s',
             ha='center', va='bottom', fontweight='bold')

# 4. 聚类分布统计
ax9 = plt.subplot(3, 3, 9)
kmeans_dist = np.bincount(best_kmeans['labels'])
ax9.bar(range(len(kmeans_dist)), kmeans_dist, color='steelblue', edgecolor='black', alpha=0.7)
ax9.set_xlabel('聚类ID', fontweight='bold')
ax9.set_ylabel('文档数量', fontweight='bold')
ax9.set_title(f'KMeans聚类分布 (K={best_kmeans["k"]})', fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 对比图表已保存: clustering_comparison.png")

# ===========================
# 7. 保存结果
# ===========================
print("\n[步骤7] 保存分析结果")
print("-"*80)

# 保存KMeans结果
kmeans_df = pd.DataFrame([{
    'K值': r['k'],
    '轮廓系数': r['silhouette'],
    'Calinski-Harabasz指数': r['calinski'],
    'Davies-Bouldin指数': r['davies'],
    '惯性': r['inertia'],
    '训练时间(秒)': r['time']
} for r in kmeans_results])
kmeans_df.to_csv('kmeans_results.csv', index=False, encoding='utf-8-sig')
print("✓ KMeans结果已保存: kmeans_results.csv")

# 保存Single-pass结果
sp_df = pd.DataFrame([{
    '阈值': r['threshold'],
    '聚类数量': r['n_clusters'],
    '轮廓系数': r['silhouette'],
    'Calinski-Harabasz指数': r['calinski'],
    'Davies-Bouldin指数': r['davies'],
    '训练时间(秒)': r['time']
} for r in singlepass_results])
sp_df.to_csv('singlepass_results.csv', index=False, encoding='utf-8-sig')
print("✓ Single-pass结果已保存: singlepass_results.csv")

# 保存最佳模型的聚类标签
result_df = df.copy()
result_df['kmeans_cluster'] = best_kmeans['labels']
result_df['singlepass_cluster'] = best_sp['labels']
result_df.to_csv('clustering_labels.csv', index=False, encoding='utf-8-sig')
print("✓ 聚类标签已保存: clustering_labels.csv")

# 保存对比总结
summary = f"""
文本聚类算法对比分析报告
{"="*80}

1. 数据概况
   - 文档数量: {len(bow_vectors):,}
   - 词汇表大小: {n_features:,}
   - 矩阵稀疏度: {sparsity:.2%}

2. KMeans 最佳结果
   - K值: {best_kmeans['k']}
   - 轮廓系数: {best_kmeans['silhouette']:.4f}
   - Calinski-Harabasz指数: {best_kmeans['calinski']:.2f}
   - Davies-Bouldin指数: {best_kmeans['davies']:.4f}
   - 惯性: {best_kmeans['inertia']:.2f}
   - 训练时间: {best_kmeans['time']:.2f}秒

3. Single-pass 最佳结果
   - 阈值: {best_sp['threshold']}
   - 聚类数量: {best_sp['n_clusters']}
   - 轮廓系数: {best_sp['silhouette']:.4f}
   - Calinski-Harabasz指数: {best_sp['calinski']:.2f}
   - Davies-Bouldin指数: {best_sp['davies']:.4f}
   - 训练时间: {best_sp['time']:.2f}秒

4. 结论
   推荐算法: {winner}
   理由: 基于轮廓系数等综合评估指标，{winner}算法在该数据集上表现更优。

   KMeans优势:
   - 聚类质量稳定
   - 聚类数量可控
   - 适合批量处理

   Single-pass优势:
   - 在线处理能力
   - 自动确定聚类数
   - 处理速度较快

5. 建议
   根据分析结果，建议使用{winner}算法进行后续的信息挖掘工作。
"""

with open('clustering_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("✓ 对比总结已保存: clustering_summary.txt")

print("\n" + "="*80)
print("✅ 分析完成！")
print("="*80)
print(f"\n🏆 最终选择: {winner}算法")
print(f"📁 生成文件:")
print(f"   1. clustering_comparison.png - 综合对比图表")
print(f"   2. kmeans_results.csv - KMeans详细结果")
print(f"   3. singlepass_results.csv - Single-pass详细结果")
print(f"   4. clustering_labels.csv - 聚类标签结果")
print(f"   5. clustering_summary.txt - 分析总结报告")
print("="*80)

plt.show()
