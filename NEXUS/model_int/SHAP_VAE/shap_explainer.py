import shap
import torch
import pickle
import numpy as np
import pandas as pd
import scanpy as sc

import seaborn as sns
from matplotlib import patches
import matplotlib.pyplot as plt
#from scipy import stats
from scipy.stats import pearsonr
from itertools import combinations
from statannotations.Annotator import Annotator


def shap_nexus(model_encoder, adata_path, output_path, device, cell_num=5000):
    adata = sc.read_h5ad(adata_path)
    random_cell_indices = np.random.choice(adata.obs.index, size=cell_num, replace=False)
    adata_shap = adata[random_cell_indices, :].copy()
    adata_shap.write(output_path+'shap_adata.h5ad')
    shap_data = torch.tensor(adata_shap.X).float()
    shap_data = shap_data.to(device)
    explainer = shap.DeepExplainer(model_encoder, shap_data)
    shap_values = explainer.shap_values(shap_data, check_additivity=False)
    with open(output_path+"shap_values.pickle", "wb") as file:
        pickle.dump(shap_values, file)

def plot_shap_selected_genes(shap_values, adata, feature_num, selected_genes, save=None):
    """
    只绘制选中基因的 shap.summary_plot，使用红蓝色渐变反映表达高低。
    """
    all_genes = list(adata.var_names)

    # 确保基因在 adata 中
    selected_genes = [g for g in selected_genes if g in all_genes]
    if not selected_genes:
        print("没有选中的基因在 adata.var_names 中，请检查基因名是否一致。")
        return

    gene_indices = [all_genes.index(g) for g in selected_genes]
    shap_values_sub = shap_values[:, gene_indices, feature_num]
    feature_values_sub = adata.X[:, gene_indices]  # 用于决定点的颜色

    # 绘图
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_sub,
                      features=feature_values_sub,
                      feature_names=selected_genes,
                      max_display=len(selected_genes),
                      show=False,
                      alpha=0.6,
                      cmap='seismic')  # 红蓝渐变

    plt.title(f"SHAP Summary Plot - Feature {feature_num}")
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
    plt.show()


def classify_shap_gene_correlations_from_adata(
    adata,
    shap_values: np.ndarray,
    key_gene_list: list,
    latent_idx: int
):
    """
    给定 AnnData 对象和 SHAP 三维数组，判断 key_gene_list 中每个基因在表达值和 SHAP 值之间的正负相关
    """
    gene_names = list(adata.var_names)
    gene_to_index = {gene: i for i, gene in enumerate(gene_names)}

    # 将表达矩阵转为 DataFrame，便于基因名访问
    X_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        columns=adata.var_names
    )

    positive_genes = []
    negative_genes = []
    correlation_dict = {}

    for gene in key_gene_list:
        if gene not in gene_to_index:
            continue

        gene_idx = gene_to_index[gene]
        expr_values = X_df[gene].values  # shape: (n_samples,)
        shap_vals = shap_values[:, gene_idx, latent_idx]  # shape: (n_samples,)

        r, p = pearsonr(expr_values, shap_vals)
        correlation_dict[gene] = r

        if r > 0:
            positive_genes.append(gene)
        elif r < 0:
            negative_genes.append(gene)

    return positive_genes, negative_genes, correlation_dict


def generate_shap_importances(adata, shap_values, save_path=None):
    gene_names = adata.var_names
    shap_importances = pd.DataFrame({'Feature': gene_names})
    num_dimensions = shap_values.shape[2]
    for i in range(num_dimensions):
        shap_importance = np.abs(shap_values[:, :, i]).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': gene_names,
            f'Importance_{i}': shap_importance
        })
        shap_importances = pd.merge(
            shap_importances, 
            importance_df, 
            on='Feature', 
            how='inner'
        )
    shap_importances = shap_importances.set_index('Feature')
    if save_path is not None:
        shap_importances.to_csv(save_path)
    return shap_importances


def get_gene_list(adata, shap_values, latent_idx, threshold = 0.01):

    shap_importances = generate_shap_importances(
        adata,
        shap_values
        )

    shap_abs = shap_importances.abs()
    total_abs_shap = shap_abs.sum(axis=1)
    contrib_ratio_abs = shap_abs.div(total_abs_shap, axis=0)
    specific_genes = {}
    for lf in shap_abs.columns:
        specific_genes[lf] = contrib_ratio_abs[contrib_ratio_abs[lf] > threshold].index.tolist()

    target_feature = 'Importance_' + str(latent_idx)  # 你关注的潜变量
    specific_genes_for_feature = specific_genes[target_feature]
    shap_values_for_feature = shap_importances.loc[specific_genes_for_feature, target_feature]
    shap_sorted_genes = shap_values_for_feature.sort_values(ascending=False)

    pos_genes, neg_genes, corrs = classify_shap_gene_correlations_from_adata(
        adata=adata,
        shap_values=shap_values,
        key_gene_list=shap_sorted_genes.index.astype(str).tolist(),
        latent_idx=latent_idx
    )

    return pos_genes, neg_genes


def plot_enrich_show(show_rows, save_path, figsize = (10,8)):
    top_go = show_rows.sort_values("Adjusted P-value").head(10)
    # 设置画布
    fig, ax = plt.subplots(figsize=figsize)
    # 提取Adjusted P-value并计算负对数（-log10(P)，值越大显著性越高）
    p_values = top_go["Adjusted P-value"]
    neg_log_p = -np.log10(p_values)  
    # 归一化负对数P值，用于颜色映射
    norm = plt.Normalize(neg_log_p.min(), neg_log_p.max())
    # 使用Reds色系，负对数P值越大（显著性越高），颜色越深
    colors = plt.cm.RdBu_r(norm(neg_log_p))
    # 绘制条形图：x轴为基因数目，y轴为GO Term，颜色由负对数P值决定
    gp.plot.barplot(
        top_go,
        cutoff=150,
        column="Gene_Count",
        title="",
        color=colors,  # 使用基于P值的颜色
        show=False,  # 不自动显示，以便后续调整
        ax = ax
    )
    # 添加颜色条：表示负对数转换后的P值
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("-log10(Adjusted P-value)", rotation=270, labelpad=20)
    
    # 设置标题与轴标签
    ax.set_title("")
    ax.set_xlabel("Count")
    ax.set_ylabel("GO Term")
    
    # 调整布局并显示
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_genes_box(adata, label_id, labels, gene_list, 
                                 test_type='Mann-Whitney', figsize_per_gene=(3, 8), 
                                 palette=None, save_path=None,
                                 comparison_strategy='adjacent',
                                 n_cols=3):
    """
    绘制多个基因的表达箱线图，支持分页显示
    
    参数:
        gene_list: 基因名称列表
        figsize_per_gene: 每个基因图的大小
        n_cols: 每行显示的基因数量
    """
    # 创建子集
    adata_use = adata[adata.obs[label_id].isin(labels)].copy()
    
    n_genes = len(gene_list)
    n_rows = (n_genes + n_cols - 1) // n_cols  # 计算需要的行数
    
    # 设置颜色
    if palette is None:
        palette = sns.color_palette("husl", len(labels))
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(figsize_per_gene[0] * n_cols, 
                                    figsize_per_gene[1] * n_rows))
    
    # 如果只有一行，将axes转换为二维数组便于统一处理
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 设置统计检验对（根据比较策略）
    pairs = []
    if comparison_strategy == 'adjacent':
        for i in range(len(labels)-1):
            pairs.append((labels[i], labels[i+1]))
    elif comparison_strategy == 'all':
        pairs = list(combinations(labels, 2))
    elif comparison_strategy == 'vs_control':
        control = labels[0]
        pairs = [(control, label) for label in labels[1:]]
    
    # 为每个基因绘制箱线图
    for idx, gene_name in enumerate(gene_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # 准备绘图数据
        plot_data = []
        for label in labels:
            mask = adata_use.obs[label_id] == label
            expr = adata_use[mask, gene_name].X.flatten()
            plot_data.extend([(label, val) for val in expr])
        
        plot_df = pd.DataFrame(plot_data, columns=[label_id, 'expression'])
        
        # 绘制箱线图
        sns.boxplot(x=label_id, y='expression', data=plot_df, 
                   palette=palette, hue=label_id, legend=False, 
                   showfliers=False, ax=ax)
        
        # 添加统计显著性标注
        if len(pairs) > 0:
            try:
                annotator = Annotator(ax, pairs, data=plot_df, 
                                    x=label_id, y='expression')
                annotator.configure(test=test_type, text_format='star', 
                                  loc='inside', verbose=0)
                annotator.apply_and_annotate()
            except Exception as e:
                print(f"基因 {gene_name} 无法添加统计标注: {str(e)}")
        
        # 美化图形
        ax.set_title(gene_name, fontsize=30, pad=10)
        ax.set_xlabel('', fontsize=40)
        ax.set_ylabel('Expression Level', fontsize=20)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='right')
    
    # 隐藏多余的子图
    for idx in range(len(gene_list), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_rank_genes_as_bar(
    adata,
    key='rank_genes',
    groups=['Sertoli cells', 'Leydig cells'],
    n_genes=10,
    sharey=False,
    fontsize=20,
    save=None,
    bar_height=0.4,  # 柱子宽度
    y_spacing=0.1    # 基因间间距
):
    """
    按 scanpy 原始排序逻辑，横坐标为差异基因的 score（评分）
    """
    # 提取差异基因数据（保留原始排序，包含 score 列）
    top_genes = []
    for group in groups:
        # 获取 scanpy 计算的 top 基因，包含 score 列
        group_df = sc.get.rank_genes_groups_df(adata, key=key, group=group).head(n_genes).copy()
        group_df['group'] = group
        top_genes.append(group_df)
    plot_df = pd.concat(top_genes, ignore_index=True)
    
    # 创建子图
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 2 + 0.25 * n_genes))
    if n_groups == 1:
        axes = [axes]
    
    # 绘制柱状图（横坐标为 score，遵循原始排序）
    light_blue = '#87CEEB'
    for i, group in enumerate(groups):
        sub_df = plot_df[plot_df['group'] == group]
        sub_df = sub_df.iloc[::-1]  # 与 scanpy 点图顺序一致（top1在最上方）
        
        # 横坐标使用 score 列
        axes[i].barh(
            sub_df['names'], 
            sub_df['scores'],  # 替换为 score 作为横坐标值
            color=light_blue,
            height=bar_height
        )
        
        # 样式设置（横坐标标签改为 score）
        axes[i].set_title(f'{group}', fontsize=fontsize)
        axes[i].set_xlabel('Score', fontsize=fontsize - 2)  # 横坐标标签改为 Score
        axes[i].tick_params(axis='y', labelsize=fontsize - 4)
        axes[i].tick_params(axis='x', labelsize=fontsize - 4)
        axes[i].axvline(x=0, color='black', linestyle='--', linewidth=1)  # score=0 参考线
        axes[i].yaxis.set_tick_params(pad=y_spacing)
        axes[i].margins(y=y_spacing)
    
    plt.subplots_adjust(wspace=0.3, hspace=y_spacing)
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"图片已保存为: {filename}")
    plt.show()