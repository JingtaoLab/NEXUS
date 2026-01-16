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
    Plot shap.summary_plot only for selected genes, using red-blue gradient to reflect expression levels.
    """
    all_genes = list(adata.var_names)

    # Ensure genes are in adata
    selected_genes = [g for g in selected_genes if g in all_genes]
    if not selected_genes:
        print("No selected genes found in adata.var_names, please check if gene names are consistent.")
        return

    gene_indices = [all_genes.index(g) for g in selected_genes]
    shap_values_sub = shap_values[:, gene_indices, feature_num]
    feature_values_sub = adata.X[:, gene_indices]  # Used to determine point colors

    # Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_sub,
                      features=feature_values_sub,
                      feature_names=selected_genes,
                      max_display=len(selected_genes),
                      show=False,
                      alpha=0.6,
                      cmap='seismic')  # Red-blue gradient

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
    Given AnnData object and SHAP 3D array, determine positive/negative correlation between expression values and SHAP values for each gene in key_gene_list
    """
    gene_names = list(adata.var_names)
    gene_to_index = {gene: i for i, gene in enumerate(gene_names)}

    # Convert expression matrix to DataFrame for easier gene name access
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

    target_feature = 'Importance_' + str(latent_idx)  # The latent variable you are interested in
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
    # Set up canvas
    fig, ax = plt.subplots(figsize=figsize)
    # Extract Adjusted P-value and calculate negative logarithm (-log10(P), larger value indicates higher significance)
    p_values = top_go["Adjusted P-value"]
    neg_log_p = -np.log10(p_values)  
    # Normalize negative log P-values for color mapping
    norm = plt.Normalize(neg_log_p.min(), neg_log_p.max())
    # Use Reds colormap, larger negative log P-value (higher significance) results in darker color
    colors = plt.cm.RdBu_r(norm(neg_log_p))
    # Plot bar chart: x-axis is gene count, y-axis is GO Term, color determined by negative log P-value
    gp.plot.barplot(
        top_go,
        cutoff=150,
        column="Gene_Count",
        title="",
        color=colors,  # Use P-value based colors
        show=False,  # Don't auto-display for subsequent adjustments
        ax = ax
    )
    # Add colorbar: represents P-values after negative log transformation
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("-log10(Adjusted P-value)", rotation=270, labelpad=20)
    
    # Set title and axis labels
    ax.set_title("")
    ax.set_xlabel("Count")
    ax.set_ylabel("GO Term")
    
    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_genes_box(adata, label_id, labels, gene_list, 
                                 test_type='Mann-Whitney', figsize_per_gene=(3, 8), 
                                 palette=None, save_path=None,
                                 comparison_strategy='adjacent',
                                 n_cols=3):
    """
    Plot expression boxplots for multiple genes, supports pagination
    
    Parameters:
        gene_list: List of gene names
        figsize_per_gene: Size of each gene plot
        n_cols: Number of genes displayed per row
    """
    # Create subset
    adata_use = adata[adata.obs[label_id].isin(labels)].copy()
    
    n_genes = len(gene_list)
    n_rows = (n_genes + n_cols - 1) // n_cols  # Calculate required number of rows
    
    # Set colors
    if palette is None:
        palette = sns.color_palette("husl", len(labels))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(figsize_per_gene[0] * n_cols, 
                                    figsize_per_gene[1] * n_rows))
    
    # If only one row, convert axes to 2D array for unified processing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Set statistical test pairs (according to comparison strategy)
    pairs = []
    if comparison_strategy == 'adjacent':
        for i in range(len(labels)-1):
            pairs.append((labels[i], labels[i+1]))
    elif comparison_strategy == 'all':
        pairs = list(combinations(labels, 2))
    elif comparison_strategy == 'vs_control':
        control = labels[0]
        pairs = [(control, label) for label in labels[1:]]
    
    # Plot boxplot for each gene
    for idx, gene_name in enumerate(gene_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Prepare plotting data
        plot_data = []
        for label in labels:
            mask = adata_use.obs[label_id] == label
            expr = adata_use[mask, gene_name].X.flatten()
            plot_data.extend([(label, val) for val in expr])
        
        plot_df = pd.DataFrame(plot_data, columns=[label_id, 'expression'])
        
        # Plot boxplot
        sns.boxplot(x=label_id, y='expression', data=plot_df, 
                   palette=palette, hue=label_id, legend=False, 
                   showfliers=False, ax=ax)
        
        # Add statistical significance annotations
        if len(pairs) > 0:
            try:
                annotator = Annotator(ax, pairs, data=plot_df, 
                                    x=label_id, y='expression')
                annotator.configure(test=test_type, text_format='star', 
                                  loc='inside', verbose=0)
                annotator.apply_and_annotate()
            except Exception as e:
                print(f"Gene {gene_name} cannot add statistical annotation: {str(e)}")
        
        # Beautify figure
        ax.set_title(gene_name, fontsize=30, pad=10)
        ax.set_xlabel('', fontsize=40)
        ax.set_ylabel('Expression Level', fontsize=20)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='right')
    
    # Hide extra subplots
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
    bar_height=0.4,  # Bar width
    y_spacing=0.1    # Spacing between genes
):
    """
    Follow scanpy's original sorting logic, x-axis is the score of differential genes
    """
    # Extract differential gene data (preserve original sorting, include score column)
    top_genes = []
    for group in groups:
        # Get top genes calculated by scanpy, include score column
        group_df = sc.get.rank_genes_groups_df(adata, key=key, group=group).head(n_genes).copy()
        group_df['group'] = group
        top_genes.append(group_df)
    plot_df = pd.concat(top_genes, ignore_index=True)
    
    # Create subplots
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 2 + 0.25 * n_genes))
    if n_groups == 1:
        axes = [axes]
    
    # Plot bar chart (x-axis is score, following original sorting)
    light_blue = '#87CEEB'
    for i, group in enumerate(groups):
        sub_df = plot_df[plot_df['group'] == group]
        sub_df = sub_df.iloc[::-1]  # Consistent with scanpy dot plot order (top1 at the top)
        
        # Use score column as x-axis
        axes[i].barh(
            sub_df['names'], 
            sub_df['scores'],  # Replace with score as x-axis value
            color=light_blue,
            height=bar_height
        )
        
        # Style settings (x-axis label changed to score)
        axes[i].set_title(f'{group}', fontsize=fontsize)
        axes[i].set_xlabel('Score', fontsize=fontsize - 2)  # X-axis label changed to Score
        axes[i].tick_params(axis='y', labelsize=fontsize - 4)
        axes[i].tick_params(axis='x', labelsize=fontsize - 4)
        axes[i].axvline(x=0, color='black', linestyle='--', linewidth=1)  # Reference line at score=0
        axes[i].yaxis.set_tick_params(pad=y_spacing)
        axes[i].margins(y=y_spacing)
    
    plt.subplots_adjust(wspace=0.3, hspace=y_spacing)
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"Image saved as: {save}")
    plt.show()