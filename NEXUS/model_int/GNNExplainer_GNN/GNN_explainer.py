import random
random.seed(42)

import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from torch_geometric.explain import Explainer, GNNExplainer

class GraphEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, edge_index, attr):
        out = self.encoder(x, edge_index, attr)
        return out[0]

def GNNexplainer_nexus(adata, model, graph_dataset, epochs=500, save_path=None):
    explainer = Explainer(
        model=GraphEncoderWrapper(model),
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=None,
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        )
    )
    
    node_masks_dict = {}
    for num in adata.obs['sample_num'].unique():
        explain_graphs = [data for data in graph_dataset if data.sample.item() == num]        
        data = explain_graphs[0]
        graph_name = str(num)
        explanation = explainer(data.x, data.edge_index, attr=data.edge_attr)
        node_masks_dict[graph_name] = explanation.node_mask.numpy()
    
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(node_masks_dict, f)
        print(f"All node_mask have been saved to : {save_path}")
    return node_masks_dict

def aggregate_graph_importance(node_masks, aggregation='mean'):
    node_masks_dict = node_masks
    importance_dict = {}
    for graph_name, node_mask in node_masks_dict.items():
        # node_importance
        if aggregation == 'mean':
            node_importance = np.mean(node_mask, axis=1)
        else:  # sum
            node_importance = np.sum(node_mask, axis=1)
        # feature_importance
        if aggregation == 'mean':
            feature_importance = np.mean(node_mask, axis=0)
        else:  # sum
            feature_importance = np.sum(node_mask, axis=0)
        
        importance_dict[graph_name] = {
            'node_importance': node_importance,
            'feature_importance': feature_importance  
        }
    return importance_dict


def extract_feature_importance(importance_dict):
    # 1. 提取所有样本名和特征重要性
    sample_names = []
    feature_importance_list = []
    
    for sample, imp_data in importance_dict.items():
        # 假设特征重要性存储在"feature_importance"键下
        # 若实际键名不同（如"feat_importance"），需相应修改
        feat_imp = imp_data.get("feature_importance", None)
        if feat_imp is not None:
            sample_names.append(sample)
            feature_importance_list.append(feat_imp)
    
    # 2. 确定特征数量（假设所有样本的特征数量一致）
    if not feature_importance_list:
        raise ValueError("importance_dict中未找到特征重要性数据")
    
    n_features = len(feature_importance_list[0])
    feature_names = [f"feature_{i}" for i in range(n_features)]  # 生成特征名（如feature_0, feature_1...）
    
    # 3. 构建DataFrame（行=特征，列=样本）
    # 先按"样本×特征"构建矩阵，再转置为"特征×样本"
    feat_imp_matrix = np.array(feature_importance_list)  # 形状：(n_samples, n_features)
    feat_imp_df = pd.DataFrame(
        feat_imp_matrix.T,  # 转置后形状：(n_features, n_samples)
        index=feature_names,  # 行名：特征名
        columns=sample_names  # 列名：样本名
    )
    
    return feat_imp_df


def GNNexplainer_hotplot(feature_importance, adata, labels=None, sort_by=None,
                         save=None, figsize=(12, 6)):
    df = feature_importance.T.copy()  # 每行是样本

    # 按 sort_by 排序样本
    if sort_by is not None:
        sorted_index = adata.obs.sort_values(by=sort_by).index
        df = df.loc[sorted_index]

    # 创建 row_colors
    row_colors = None
    legend_patches = []

    if labels is not None:
        row_colors_df = pd.DataFrame(index=df.index)

        # 每个 label 用不同调色板
        palettes = ['tab20', 'Set2', 'Pastel1', 'Set1', 'Dark2','tab10',  'Accent', 'Paired']
        for i, label in enumerate(labels):
            if label not in adata.obs.columns:
                continue

            unique_vals = adata.obs[label].astype(str).unique()
            palette = sns.color_palette(palettes[i % len(palettes)], len(unique_vals))
            lut = dict(zip(unique_vals, palette))
            mapped_colors = adata.obs.loc[df.index, label].astype(str).map(lut)

            row_colors_df[label] = mapped_colors

            # 图例颜色块
            legend_patches += [
                mpatches.Patch(color=lut[val], label=f"{label}: {val}") for val in unique_vals
            ]

        row_colors = row_colors_df

    # 热图：无聚类，使用颜色条
    g = sns.clustermap(df,
                       cmap="coolwarm",
                       row_colors=row_colors,
                       col_cluster=True,
                       row_cluster=True,
                       xticklabels=True,
                       yticklabels=False,
                       #standard_scale=1,
                       method='average',
                       metric='cosine',
                       cbar_pos=(1.05, 0.2, 0.03, 0.3),
                       dendrogram_ratio=(0.1, 0.1),
                       figsize=figsize)

    g.fig.suptitle("GNN Explainer Feature Importance Heatmap", y=1.02)

    # 添加图例
    if legend_patches:
        g.ax_heatmap.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.)

    # 保存或显示
    if save:
        g.savefig(save, bbox_inches="tight")

    plt.show()


