"""
为汇报创建额外的可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

print("="*60)
print("为汇报创建可视化图表")
print("="*60)

# 1. 主题分布柱状图
print("\n[1/4] 创建主题分布柱状图...")
topic_dist = {
    '主题1\n台湾历史': 18.9,
    '主题2\n中日外交': 55.7,
    '主题3\n体育抗战': 6.9,
    '主题4\n经济科技': 18.6
}

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax.bar(topic_dist.keys(), topic_dist.values(), color=colors, alpha=0.8, edgecolor='black')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('文档占比 (%)', fontsize=14, fontweight='bold')
ax.set_title('LDA主题模型 - 四个主题的分布', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 65)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加主导主题标记
ax.annotate('主导主题', xy=(1, 55.7), xytext=(1.5, 60),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('topic_distribution.png', dpi=300, bbox_inches='tight')
print("✓ 保存为: topic_distribution.png")
plt.close()

# 2. 聚类算法对比柱状图
print("\n[2/4] 创建聚类算法对比图...")
algorithms = ['KMeans\n(K=3)', 'Single-pass\n(阈值=0.7)']
silhouette = [0.8626, -0.2147]
ch_scores = [1224.22, 13.18]
db_scores = [0.2485, 1.6715]
times = [0.15, 7.69]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 轮廓系数
colors_sil = ['#2ecc71', '#e74c3c']
bars1 = ax1.bar(algorithms, silhouette, color=colors_sil, alpha=0.8, edgecolor='black')
ax1.set_ylabel('轮廓系数', fontsize=12, fontweight='bold')
ax1.set_title('轮廓系数对比 (越大越好)', fontsize=13, fontweight='bold')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax1.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=11, fontweight='bold')

# Calinski-Harabasz
bars2 = ax2.bar(algorithms, ch_scores, color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
ax2.set_ylabel('Calinski-Harabasz指数', fontsize=12, fontweight='bold')
ax2.set_title('Calinski-Harabasz指数对比 (越大越好)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Davies-Bouldin
bars3 = ax3.bar(algorithms, db_scores, color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
ax3.set_ylabel('Davies-Bouldin指数', fontsize=12, fontweight='bold')
ax3.set_title('Davies-Bouldin指数对比 (越小越好)', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 训练时间
bars4 = ax4.bar(algorithms, times, color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
ax4.set_ylabel('训练时间 (秒)', fontsize=12, fontweight='bold')
ax4.set_title('训练时间对比 (越小越好)', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('聚类算法性能对比：KMeans vs Single-pass',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('algorithm_comparison_simple.png', dpi=300, bbox_inches='tight')
print("✓ 保存为: algorithm_comparison_simple.png")
plt.close()

# 3. KMeans不同K值对比
print("\n[3/4] 创建KMeans K值选择图...")
kmeans_data = pd.read_csv('kmeans_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

k_values = kmeans_data['K值'].values
silhouette_scores = kmeans_data['轮廓系数'].values

bars = ax.bar(k_values, silhouette_scores, color='#3498db', alpha=0.8, edgecolor='black')
bars[0].set_color('#e74c3c')  # K=3 用红色标记

ax.set_xlabel('聚类数量 (K)', fontsize=14, fontweight='bold')
ax.set_ylabel('轮廓系数', fontsize=14, fontweight='bold')
ax.set_title('KMeans聚类 - 不同K值的轮廓系数', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 标记最优值
max_idx = silhouette_scores.argmax()
ax.annotate(f'最优: K={k_values[max_idx]}\n轮廓系数={silhouette_scores[max_idx]:.3f}',
            xy=(k_values[max_idx], silhouette_scores[max_idx]),
            xytext=(k_values[max_idx]+2, silhouette_scores[max_idx]-0.1),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# 添加数值标签
for i, (k, score) in enumerate(zip(k_values, silhouette_scores)):
    ax.text(k, score + 0.02, f'{score:.2f}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('kmeans_k_selection.png', dpi=300, bbox_inches='tight')
print("✓ 保存为: kmeans_k_selection.png")
plt.close()

# 4. 主题关键词Top 10表格图
print("\n[4/4] 创建主题关键词表格图...")
topic_words = pd.read_csv('lda_topic_words.csv')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

topic_names = [
    '主题1: 台湾地位与历史争议',
    '主题2: 中日外交关系与政治立场',
    '主题3: 体育赛事与抗战历史',
    '主题4: 经济科技发展与社会服务'
]

colors_topics = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for topic_id, ax in enumerate(axes, start=1):
    # 获取该主题的Top 10关键词
    topic_data = topic_words[
        (topic_words['主题编号'] == topic_id) &
        (topic_words['排名'] <= 10)
    ].sort_values('排名')

    words = topic_data['词汇'].values[::-1]  # 反转顺序，最重要的在上面
    weights = topic_data['权重'].values[::-1]

    # 归一化权重
    weights_norm = weights / weights.max() * 100

    # 绘制横向柱状图
    bars = ax.barh(range(len(words)), weights_norm,
                   color=colors_topics[topic_id-1], alpha=0.7, edgecolor='black')

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=11)
    ax.set_xlabel('相对权重', fontsize=11, fontweight='bold')
    ax.set_title(topic_names[topic_id-1], fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加数值标签
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{weight:.0f}',
                va='center', fontsize=9, color='black')

plt.suptitle('LDA主题模型 - 各主题Top 10关键词',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('topic_keywords_top10.png', dpi=300, bbox_inches='tight')
print("✓ 保存为: topic_keywords_top10.png")
plt.close()

# 5. 创建数据概况图
print("\n[5/5] 创建数据概况图...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 数据规模
data_info = ['文档数量', '词汇数', '稀疏度(%)', '非零元素(%)']
data_values = [5219, 60415, 99.84, 0.16]
colors_data = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

bars1 = ax1.bar(range(len(data_info)), data_values, color=colors_data, alpha=0.8, edgecolor='black')
ax1.set_xticks(range(len(data_info)))
ax1.set_xticklabels(data_info, fontsize=11)
ax1.set_ylabel('数值', fontsize=12, fontweight='bold')
ax1.set_title('数据集基本信息', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars1, data_values)):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:,.2f}' if i >= 2 else f'{val:,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# LDA困惑度比较（简化版）
lda_results = pd.read_csv('olda_perplexity_results.csv')
topic_nums = lda_results['主题个数'].values
perplexities = lda_results['困惑度'].values

ax2.plot(topic_nums, perplexities, marker='o', linewidth=2, markersize=8, color='#e74c3c')
ax2.scatter([4], [perplexities[1]], s=300, c='gold', marker='*',
           edgecolors='red', linewidths=2, zorder=5)
ax2.set_xlabel('主题个数', fontsize=12, fontweight='bold')
ax2.set_ylabel('困惑度', fontsize=12, fontweight='bold')
ax2.set_title('LDA困惑度分析（最优: 4个主题）', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.annotate('最优', xy=(4, perplexities[1]), xytext=(6, perplexities[1]-200),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', fontweight='bold')

# 聚类数量对比
cluster_counts = ['KMeans\n(K=3)', 'Single-pass\n(阈值=0.3)', 'Single-pass\n(阈值=0.5)', 'Single-pass\n(阈值=0.7)']
counts = [3, 65, 156, 569]
colors_cluster = ['#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c']

bars3 = ax3.bar(range(len(cluster_counts)), counts, color=colors_cluster, alpha=0.8, edgecolor='black')
ax3.set_xticks(range(len(cluster_counts)))
ax3.set_xticklabels(cluster_counts, fontsize=10)
ax3.set_ylabel('聚类数量', fontsize=12, fontweight='bold')
ax3.set_title('不同算法的聚类数量对比', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(axis='y', alpha=0.3)
for bar, count in zip(bars3, counts):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{count}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 主题分布饼图
topic_percentages = [18.9, 55.7, 6.9, 18.6]
topic_labels = ['主题1\n台湾历史\n18.9%', '主题2\n中日外交\n55.7%',
                '主题3\n体育抗战\n6.9%', '主题4\n经济科技\n18.6%']
colors_pie = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

wedges, texts, autotexts = ax4.pie(topic_percentages, labels=topic_labels, colors=colors_pie,
                        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('主题分布占比', fontsize=13, fontweight='bold')

# 突出主导主题
wedges[1].set_edgecolor('red')
wedges[1].set_linewidth(3)

plt.suptitle('项目数据概况与核心结果', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('data_overview.png', dpi=300, bbox_inches='tight')
print("✓ 保存为: data_overview.png")
plt.close()

print("\n" + "="*60)
print("✅ 所有图表创建完成！")
print("="*60)
print("\n已生成以下图表：")
print("1. topic_distribution.png - 主题分布柱状图")
print("2. algorithm_comparison_simple.png - 聚类算法对比图")
print("3. kmeans_k_selection.png - KMeans K值选择图")
print("4. topic_keywords_top10.png - 主题关键词图")
print("5. data_overview.png - 数据概况与核心结果")
print("\n这些图表可以直接用于PPT汇报！")
print("="*60)
