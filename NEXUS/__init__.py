"""
NEXUS: A contrastive learning model for single-cell RNA sequencing analysis
"""

import os
import sys

# Get the package directory
_package_dir = os.path.dirname(os.path.abspath(__file__))

# Version
__version__ = "0.1.0"

# Utility functions for accessing data and tutorials
def get_test_data_path():
    """
    Get the path to the test_data directory.
    
    Returns:
        str: Path to the test_data directory
    """
    # Try to find test_data in the installed package location
    possible_paths = [
        os.path.join(_package_dir, '..', 'test_data'),
        os.path.join(_package_dir, '..', '..', 'test_data'),
        os.path.join(sys.prefix, 'nexus', 'test_data'),
        os.path.join(sys.prefix, 'Lib', 'site-packages', 'nexus', 'test_data'),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    # Fallback: return relative path from package
    return os.path.join(_package_dir, '..', 'test_data')

def get_tutorial_path():
    """
    Get the path to the tutorial directory.
    
    Returns:
        str: Path to the tutorial directory
    """
    # Try to find tutorial in the installed package location
    possible_paths = [
        os.path.join(_package_dir, '..', 'tutorial'),
        os.path.join(_package_dir, '..', '..', 'tutorial'),
        os.path.join(sys.prefix, 'nexus', 'tutorial'),
        os.path.join(sys.prefix, 'Lib', 'site-packages', 'nexus', 'tutorial'),
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
    
    # Fallback: return relative path from package
    return os.path.join(_package_dir, '..', 'tutorial')

# Import commonly used classes and functions for convenience
# Users can still use: from NEXUS.cell_emb.model import OneViewVAE
# Or use the direct imports below after installation

try:
    # Cell embedding models
    from NEXUS.cell_emb.model import (
        OneViewVAE,
        TwoViewVAE,
        EncoderOnly,
        train_model,
        test_model,
        embedding,
        loss_function_oneview,
        loss_function_twoview,
    )
    
    # Data processing
    from NEXUS.data_process.data_preprocess_cell import Data_process
    from NEXUS.data_process.data_process_sample import (
        Adata_to_graph,
        creat_adata_graph,
        creat_graph_adata,
    )
    
    # Sample embedding models
    from NEXUS.sample_emb.Graph_model import (
        GraphVAE,
        train_model as train_graph_model,
        embedding as graph_embedding,
        loss_function as graph_loss_function,
    )
    
    # Plotting utilities
    from NEXUS.plotting.emb_plot import (
        cell_embedding_umap_plot,
        sample_embedding_umap_plot,
    )
    
    # Model interpretation
    from NEXUS.model_int.SHAP_VAE.shap_explainer import (
        shap_nexus,
        classify_shap_gene_correlations_from_adata,
        generate_shap_importances,
        get_gene_list,
        plot_shap_selected_genes,
        plot_enrich_show,
        plot_genes_box,
    )
    
    from NEXUS.model_int.GNNExplainer_GNN.GNN_explainer import (
        GraphEncoderWrapper,
        GNNexplainer_nexus,
        aggregate_graph_importance,
        extract_feature_importance,
        GNNexplainer_hotplot,
    )
    
except ImportError:
    # If imports fail (e.g., during package installation), just pass
    # Users can still import directly from submodules
    pass

__all__ = [
    # Version
    '__version__',
    
    # Utility functions
    'get_test_data_path',
    'get_tutorial_path',
    
    # Cell embedding
    'OneViewVAE',
    'TwoViewVAE',
    'EncoderOnly',
    'train_model',
    'test_model',
    'embedding',
    'loss_function_oneview',
    'loss_function_twoview',
    
    # Data processing
    'Data_process',
    'Adata_to_graph',
    'creat_adata_graph',
    'creat_graph_adata',
    
    # Sample embedding
    'GraphVAE',
    'train_graph_model',
    'graph_embedding',
    'graph_loss_function',
    
    # Plotting
    'cell_embedding_umap_plot',
    'sample_embedding_umap_plot',
    
    # SHAP
    'shap_nexus',
    'classify_shap_gene_correlations_from_adata',
    'generate_shap_importances',
    'get_gene_list',
    'plot_shap_selected_genes',
    'plot_enrich_show',
    'plot_genes_box',
    
    # GNN Explainer
    'GraphEncoderWrapper',
    'GNNexplainer_nexus',
    'aggregate_graph_importance',
    'extract_feature_importance',
    'GNNexplainer_hotplot',
]

