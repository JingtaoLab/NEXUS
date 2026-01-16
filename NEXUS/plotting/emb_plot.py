import numpy as np
random_seed = 42
np.random.seed(random_seed)
import scanpy as sc
import harmonypy as harmony

def cell_embedding_umap_plot(adata, label, batch = None, harm = False, resolution=1.0):
    sc.pp.neighbors(adata, use_rep="X_VAE")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)
    adata.obs["leiden_emb"] = adata.obs["leiden"]
    adata.obsm["umap_emb"] = adata.obsm["X_umap"]
    sc.pl.umap(adata, color = label, size = 5, title = '', 
               save="_cell_embedding.png"
              )
    if batch is not None:
        if harm is True:
            VAE_harmony = harmony.run_harmony(adata.obsm["X_VAE"], adata.obs, [batch])
            adata.obsm['X_VAE_harmony'] = VAE_harmony.Z_corr.T
            sc.pp.neighbors(adata, use_rep='X_VAE_harmony')
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=1.0)
            adata.obs["leiden_harmony"] = adata.obs["leiden"]
            adata.obsm["umap_harmony"] = adata.obsm["X_umap"]
            sc.pl.umap(adata, color = label, size = 5, title = '', 
                       save="_cell_embedding_harmony.png"
                       )
    return adata

def sample_embedding_umap_plot(adata, labels, n_neighbors = 5):
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors, use_rep="X_pca")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0)
    adata.obs["leiden_embedding"] = adata.obs["leiden"]
    adata.obsm["NEXUS_umap"] = adata.obsm["X_umap"]
    for label in labels:
        sc.pl.umap(adata, color = label, size = 300, title = '', 
                   save="_NEXUS_" + label + ".png"
                  )
    return adata

