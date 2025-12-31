import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("在线LDA (OLDA) 按周分段主题建模分析")
print("="*80)

# ===========================
# 1. 数据加载和预处理
# ===========================
print("\n[步骤1] 数据加载和预处理")
print("-"*80)
start_time = time.time()

df = pd.read_csv('cn_bow.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"✓ 数据读取完成")
print(f"  - 总文档数: {len(df):,}")
print(f"  - 日期范围: {df['date'].min()} 至 {df['date'].max()}")

# 计算总天数并按周分段
min_date = df['date'].min()
max_date = df['date'].max()
total_days = (max_date - min_date).days + 1
print(f"  - 总天数: {total_days}")

# 创建4周的日期范围
week_ranges = []
start_date = min_date
for i in range(4):
    end_date = start_date + timedelta(days=7)
    if end_date > max_date:
        end_date = max_date
    week_ranges.append((start_date, end_date, i+1))
    start_date = end_date

print(f"\n  按周划分:")
for start, end, week_num in week_ranges:
    week_data = df[(df['date'] >= start) & (df['date'] < end)]
    print(f"    第{week_num}周: {start.date()} 至 {end.date()} ({len(week_data)} 文档)")

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
idx_to_word_id = {idx: word_id for word_id, idx in word_id_to_idx.items()}
n_features = len(all_word_ids)

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
# 2. 按周进行OLDA主题建模
# ===========================
print("\n[步骤2] 按周进行OLDA主题建模")
print("="*80)

# 参数设置
n_topics_list = [3, 5, 8, 10]  # 测试不同的主题数
best_n_topics = 5  # 默认最佳主题数

weekly_results = []

for start_date, end_date, week_num in week_ranges:
    print(f"\n{'='*80}")
    print(f"第{week_num}周分析: {start_date.date()} 至 {end_date.date()}")
    print(f"{'='*80}")

    # 筛选该周的数据
    week_mask = (df['date'] >= start_date) & (df['date'] < end_date)
    week_indices = df[week_mask].index.tolist()
    X_week = X[week_indices]

    print(f"  文档数: {len(week_indices)}")

    # 测试不同主题数
    week_topic_results = []
    for n_topics in n_topics_list:
        print(f"\n  测试主题数: {n_topics}")
        start_time = time.time()

        # 使用在线LDA
        olda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='online',
            batch_size=128,
            max_iter=50,
            learning_offset=50.,
            random_state=42,
            n_jobs=-1
        )

        doc_topic_dist = olda.fit_transform(X_week)
        fit_time = time.time() - start_time

        # 计算困惑度
        perplexity = olda.perplexity(X_week)

        # 计算主题一致性（使用轮廓系数作为替代指标）
        topic_labels = doc_topic_dist.argmax(axis=1)
        if len(set(topic_labels)) > 1:
            silhouette = silhouette_score(doc_topic_dist, topic_labels)
        else:
            silhouette = -1

        week_topic_results.append({
            'n_topics': n_topics,
            'perplexity': perplexity,
            'silhouette': silhouette,
            'time': fit_time,
            'model': olda,
            'doc_topic_dist': doc_topic_dist
        })

        print(f"    - 困惑度: {perplexity:.2f}")
        print(f"    - 轮廓系数: {silhouette:.4f}")
        print(f"    - 训练时间: {fit_time:.2f}秒")

    # 选择最佳主题数（困惑度最低）
    best_result = min(week_topic_results, key=lambda x: x['perplexity'])
    print(f"\n  ✓ 最佳主题数: {best_result['n_topics']} (困惑度: {best_result['perplexity']:.2f})")

    # 提取主题词
    print(f"\n  主题关键词 (Top 10):")
    feature_names = [idx_to_word_id[i] for i in range(n_features)]
    n_top_words = 10

    topic_keywords = []
    for topic_idx, topic in enumerate(best_result['model'].components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [str(feature_names[i]) for i in top_indices]
        top_weights = [topic[i] for i in top_indices]

        print(f"    主题 {topic_idx}: {' '.join(top_words[:5])}")

        topic_keywords.append({
            'week': week_num,
            'topic': topic_idx,
            'keywords': ' '.join(top_words),
            'weights': ' '.join([f'{w:.4f}' for w in top_weights])
        })

    # 保存该周结果
    weekly_results.append({
        'week_num': week_num,
        'start_date': start_date,
        'end_date': end_date,
        'n_docs': len(week_indices),
        'best_n_topics': best_result['n_topics'],
        'perplexity': best_result['perplexity'],
        'silhouette': best_result['silhouette'],
        'time': best_result['time'],
        'model': best_result['model'],
        'doc_topic_dist': best_result['doc_topic_dist'],
        'topic_keywords': topic_keywords,
        'all_results': week_topic_results,
        'week_indices': week_indices
    })

# ===========================
# 3. 结果汇总和可视化
# ===========================
print("\n" + "="*80)
print("[步骤3] 结果汇总和可视化")
print("="*80)

# 创建综合对比图
fig = plt.figure(figsize=(20, 12))

# 1. 每周最佳困惑度对比
ax1 = plt.subplot(2, 3, 1)
weeks = [r['week_num'] for r in weekly_results]
perplexities = [r['perplexity'] for r in weekly_results]
ax1.bar(weeks, perplexities, color='steelblue', edgecolor='black', linewidth=2)
ax1.set_xlabel('周次', fontweight='bold', fontsize=12)
ax1.set_ylabel('困惑度', fontweight='bold', fontsize=12)
ax1.set_title('各周OLDA困惑度对比', fontweight='bold', fontsize=14)
ax1.set_xticks(weeks)
ax1.set_xticklabels([f'第{w}周' for w in weeks])
ax1.grid(True, alpha=0.3, axis='y')
for i, (w, p) in enumerate(zip(weeks, perplexities)):
    ax1.text(w, p + max(perplexities)*0.02, f'{p:.1f}', ha='center', va='bottom', fontweight='bold')

# 2. 每周文档数量
ax2 = plt.subplot(2, 3, 2)
doc_counts = [r['n_docs'] for r in weekly_results]
ax2.bar(weeks, doc_counts, color='lightcoral', edgecolor='black', linewidth=2)
ax2.set_xlabel('周次', fontweight='bold', fontsize=12)
ax2.set_ylabel('文档数量', fontweight='bold', fontsize=12)
ax2.set_title('各周文档数量分布', fontweight='bold', fontsize=14)
ax2.set_xticks(weeks)
ax2.set_xticklabels([f'第{w}周' for w in weeks])
ax2.grid(True, alpha=0.3, axis='y')
for i, (w, d) in enumerate(zip(weeks, doc_counts)):
    ax2.text(w, d + max(doc_counts)*0.02, f'{d}', ha='center', va='bottom', fontweight='bold')

# 3. 每周最佳主题数
ax3 = plt.subplot(2, 3, 3)
best_topics = [r['best_n_topics'] for r in weekly_results]
ax3.bar(weeks, best_topics, color='lightgreen', edgecolor='black', linewidth=2)
ax3.set_xlabel('周次', fontweight='bold', fontsize=12)
ax3.set_ylabel('最佳主题数', fontweight='bold', fontsize=12)
ax3.set_title('各周最佳主题数', fontweight='bold', fontsize=14)
ax3.set_xticks(weeks)
ax3.set_xticklabels([f'第{w}周' for w in weeks])
ax3.set_yticks(range(min(best_topics), max(best_topics)+1))
ax3.grid(True, alpha=0.3, axis='y')
for i, (w, t) in enumerate(zip(weeks, best_topics)):
    ax3.text(w, t + 0.1, f'{t}', ha='center', va='bottom', fontweight='bold')

# 4. 训练时间对比
ax4 = plt.subplot(2, 3, 4)
times = [r['time'] for r in weekly_results]
ax4.bar(weeks, times, color='gold', edgecolor='black', linewidth=2)
ax4.set_xlabel('周次', fontweight='bold', fontsize=12)
ax4.set_ylabel('训练时间 (秒)', fontweight='bold', fontsize=12)
ax4.set_title('各周OLDA训练时间', fontweight='bold', fontsize=14)
ax4.set_xticks(weeks)
ax4.set_xticklabels([f'第{w}周' for w in weeks])
ax4.grid(True, alpha=0.3, axis='y')
for i, (w, t) in enumerate(zip(weeks, times)):
    ax4.text(w, t + max(times)*0.02, f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')

# 5. 困惑度随主题数变化（选取第1周作为示例）
ax5 = plt.subplot(2, 3, 5)
week1_results = weekly_results[0]['all_results']
n_topics_tested = [r['n_topics'] for r in week1_results]
perplexities_tested = [r['perplexity'] for r in week1_results]
ax5.plot(n_topics_tested, perplexities_tested, 'o-', linewidth=2, markersize=8, color='purple')
ax5.scatter([weekly_results[0]['best_n_topics']], [weekly_results[0]['perplexity']],
           color='red', s=200, marker='*', zorder=5, label='最佳')
ax5.set_xlabel('主题数', fontweight='bold', fontsize=12)
ax5.set_ylabel('困惑度', fontweight='bold', fontsize=12)
ax5.set_title('第1周: 主题数 vs 困惑度', fontweight='bold', fontsize=14)
ax5.grid(True, alpha=0.3)
ax5.legend()

# 6. 主题分布热力图（第1周）
ax6 = plt.subplot(2, 3, 6)
doc_topic = weekly_results[0]['doc_topic_dist']
# 只显示前50个文档的主题分布
if len(doc_topic) > 50:
    doc_topic_display = doc_topic[:50]
else:
    doc_topic_display = doc_topic
im = ax6.imshow(doc_topic_display.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax6.set_xlabel('文档索引', fontweight='bold', fontsize=12)
ax6.set_ylabel('主题', fontweight='bold', fontsize=12)
ax6.set_title(f'第1周: 文档-主题分布热力图 (前{len(doc_topic_display)}文档)', fontweight='bold', fontsize=14)
plt.colorbar(im, ax=ax6, label='主题概率')

plt.tight_layout()
plt.savefig('olda_weekly_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 可视化图表已保存: olda_weekly_analysis.png")

# ===========================
# 4. 保存结果
# ===========================
print("\n[步骤4] 保存分析结果")
print("-"*80)

# 保存每周主题关键词
all_keywords = []
for result in weekly_results:
    all_keywords.extend(result['topic_keywords'])
keywords_df = pd.DataFrame(all_keywords)
keywords_df.to_csv('olda_weekly_topic_keywords.csv', index=False, encoding='utf-8-sig')
print("✓ 每周主题关键词已保存: olda_weekly_topic_keywords.csv")

# 保存每周汇总结果
summary_data = []
for result in weekly_results:
    summary_data.append({
        '周次': f'第{result["week_num"]}周',
        '开始日期': result['start_date'].date(),
        '结束日期': result['end_date'].date(),
        '文档数': result['n_docs'],
        '最佳主题数': result['best_n_topics'],
        '困惑度': result['perplexity'],
        '轮廓系数': result['silhouette'],
        '训练时间(秒)': result['time']
    })
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('olda_weekly_summary.csv', index=False, encoding='utf-8-sig')
print("✓ 每周汇总结果已保存: olda_weekly_summary.csv")

# 保存文档的主题分配
doc_topics_data = []
for result in weekly_results:
    week_num = result['week_num']
    doc_topic_dist = result['doc_topic_dist']
    week_indices = result['week_indices']

    for local_idx, global_idx in enumerate(week_indices):
        topic_probs = doc_topic_dist[local_idx]
        dominant_topic = topic_probs.argmax()
        doc_topics_data.append({
            '周次': week_num,
            '文档索引': global_idx,
            '日期': df.loc[global_idx, 'date'],
            '标签': df.loc[global_idx, 'label'],
            '主要主题': dominant_topic,
            '主题概率': topic_probs[dominant_topic],
            **{f'主题{i}概率': topic_probs[i] for i in range(len(topic_probs))}
        })

doc_topics_df = pd.DataFrame(doc_topics_data)
doc_topics_df.to_csv('olda_weekly_document_topics.csv', index=False, encoding='utf-8-sig')
print("✓ 文档主题分配已保存: olda_weekly_document_topics.csv")

# 生成分析报告
report = f"""
在线LDA (OLDA) 按周分段主题建模分析报告
{"="*80}

1. 数据概况
   - 总文档数: {len(df):,}
   - 日期范围: {df['date'].min()} 至 {df['date'].max()}
   - 总天数: {total_days}
   - 词汇表大小: {n_features:,}
   - 矩阵稀疏度: {sparsity:.2%}

2. 分周分析结果
"""

for result in weekly_results:
    report += f"""
   第{result['week_num']}周 ({result['start_date'].date()} 至 {result['end_date'].date()})
   -----------------------------------------
   - 文档数: {result['n_docs']:,}
   - 最佳主题数: {result['best_n_topics']}
   - 困惑度: {result['perplexity']:.2f}
   - 轮廓系数: {result['silhouette']:.4f}
   - 训练时间: {result['time']:.2f}秒

   主题关键词:
"""
    for topic_kw in result['topic_keywords']:
        keywords = ' '.join(topic_kw['keywords'].split()[:5])
        report += f"      主题{topic_kw['topic']}: {keywords}\n"

report += f"""

3. 整体趋势分析
   - 平均困惑度: {np.mean(perplexities):.2f}
   - 困惑度标准差: {np.std(perplexities):.2f}
   - 平均主题数: {np.mean(best_topics):.1f}
   - 总训练时间: {sum(times):.2f}秒

4. 主要发现
   - 困惑度最低的周: 第{perplexities.index(min(perplexities)) + 1}周 (困惑度: {min(perplexities):.2f})
   - 文档数最多的周: 第{doc_counts.index(max(doc_counts)) + 1}周 ({max(doc_counts)} 文档)
   - 主题数最多的周: 第{best_topics.index(max(best_topics)) + 1}周 ({max(best_topics)} 主题)

5. 建议
   - OLDA模型能够有效地处理各周数据的主题建模
   - 不同周的主题数存在差异，反映了内容的多样性变化
   - 建议根据具体周的主题关键词进行深入的内容分析

6. 输出文件
   - olda_weekly_analysis.png: 各周OLDA分析可视化
   - olda_weekly_topic_keywords.csv: 每周主题关键词
   - olda_weekly_summary.csv: 每周汇总统计
   - olda_weekly_document_topics.csv: 文档级主题分配
   - olda_weekly_report.txt: 本分析报告
"""

with open('olda_weekly_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✓ 分析报告已保存: olda_weekly_report.txt")

print("\n" + "="*80)
print("✅ OLDA按周分段分析完成！")
print("="*80)
print(f"\n📁 生成文件:")
print(f"   1. olda_weekly_analysis.png - 各周分析可视化")
print(f"   2. olda_weekly_topic_keywords.csv - 每周主题关键词")
print(f"   3. olda_weekly_summary.csv - 每周汇总统计")
print(f"   4. olda_weekly_document_topics.csv - 文档主题分配")
print(f"   5. olda_weekly_report.txt - 详细分析报告")
print("="*80)

# 打印简要汇总
print("\n【各周简要总结】")
for result in weekly_results:
    print(f"  第{result['week_num']}周: {result['n_docs']:4d}文档, {result['best_n_topics']}主题, 困惑度{result['perplexity']:6.1f}")

plt.show()
