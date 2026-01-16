import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import random
import pickle
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
#import networkx as nx

from scipy.spatial.distance import cdist

def create_adjacency_matrix(coordinates, k):
    """
    Efficiently compute distances between cells using cdist and generate adjacency matrix.
    """
    # Calculate Euclidean distances between all cell pairs
    distance_matrix = cdist(coordinates, coordinates, metric='euclidean')

    # Create adjacency matrix: 1 when distance is less than or equal to k, 0 otherwise
    adj_matrix = (distance_matrix <= k).astype(int)

    # Since adjacency matrix is symmetric, ensure diagonal is 0 (no self-connections)
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix

class Adata_to_graph:
    def __init__(self, adata, split_ratio, batch_size):
        ''''''
        self.adata = adata
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        umap_coords = adata.obsm["X_umap"]
        scaler = MinMaxScaler()
        umap_coords_scaled = scaler.fit_transform(umap_coords)
        adata.obsm["X_umap_norm"] = umap_coords_scaled
        
    def label_meta(self):
        for label in ['sample', 'cluster']:
            unique_clusters = self.adata.obs[label].unique()
            cluster_dict = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
            # Use map function to apply cluster_dict to obs
            self.adata.obs[label + '_num'] = self.adata.obs[label].map(cluster_dict)
        return self.adata
    
    def get_graph_dataset(self):
        samples = self.adata.obs['sample']
        unique_samples = np.unique(samples)
        # Create graph dataset
        self.graph_dataset = []

        # Iterate through each category
        for sample in unique_samples:
            # Get cell subset for current category
            cell_subset = self.adata[self.adata.obs['sample'] == sample]
            cell = cell_subset.X.toarray()

            # Build graph structure
            # Extract umap as basis for distance calculation and build adjacency matrix
            X_umap = cell_subset.obsm['X_umap_norm']
            adjacency_matrix = create_adjacency_matrix(X_umap, 0.005)

            # Create graph object
            #G = nx.from_numpy_array(adjacency_matrix)

            # Use gene expression data as node features
            x = torch.FloatTensor(cell)

            # Create edge index
            edge_index = torch.tensor(np.transpose(adjacency_matrix.nonzero()), dtype=torch.long)

            # Extract edge feature values from adjacency matrix
            edge_features = []
            for edge in edge_index:
                src, dst = edge.tolist()
                edge_features.append(adjacency_matrix[src][dst])
            # Convert edge features to tensor
            edge_attr = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)  # Assume edge features are 1D, adjust shape and dtype as needed

            # Add labels for each node
            node_index = torch.tensor(cell_subset.obs['cell_index'].to_numpy(), dtype=torch.long)
            data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y = node_index)  # Transpose to meet PyG requirements

            # Add graph labels
            for label in ['sample', 'cluster']:
                label_cluster = cell_subset.obs[label + '_num']
                label_cluster = label_cluster.to_numpy()
                y = torch.tensor(label_cluster)
                setattr(data, label, y[0])

            # Add graph data to graph dataset
            self.graph_dataset.append(data)
    
    def dataloader(self):
        random.shuffle(self.graph_dataset)
        train = int(self.split_ratio*len(self.graph_dataset))
        # Dataset splitting
        train_dataset=[]
        for graph in self.graph_dataset[:train]:
            if len(graph.x) > 1000:
                train_dataset.append(graph)
                
        test_dataset=[]
        for graph in self.graph_dataset[train:]:
            if len(graph.x) > 1000:
                test_dataset.append(graph)
        #train_dataset = self.graph_dataset[:train]
        #test_dataset = self.graph_dataset[train:]
        for i in range(len(train_dataset)):
            train_dataset[i].train_test = torch.tensor(0)
        for i in range(len(test_dataset)):
            test_dataset[i].train_test = torch.tensor(1)
        #self.graph_dataset = train_dataset + test_dataset
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return self.graph_dataset, train_loader, test_loader


def creat_adata_graph(adata, sample_col, cluster_col=None, harm=False):
    if harm is True:
        adata_graph = ad.AnnData(X=adata.obsm['umap_harmony'], obs=adata.obs, uns=adata.uns, obsm=adata.obsm)
    if harm is False:
        adata_graph = ad.AnnData(X=adata.obsm['X_VAE'], obs=adata.obs, uns=adata.uns, obsm=adata.obsm)
    adata_graph.obs = adata_graph.obs.reset_index(drop=True)
    adata_graph.obs['cell_index'] = adata_graph.obs.index
    adata_graph.obs['sample'] = adata_graph.obs[sample_col]
    if cluster_col is not None:
        adata_graph.obs['cluster'] = adata_graph.obs[cluster_col]
    else:
        adata_graph.obs['cluster'] = adata_graph.obs[sample_col]

    class_num = len(np.unique(adata_graph.obs['cluster']))
    
    return adata_graph, class_num


def explode_nested_relationships(df, main_col, other_cols):
    result_rows = []
    
    for a_value in df[main_col].unique():
        sub_df = df[df[main_col] == a_value]
        
        # Get unique values for each other_col
        value_lists = {col: sub_df[col].unique() for col in other_cols}
        
        # Construct all possible combinations (using itertools.product)
        import itertools
        combinations = list(itertools.product(*value_lists.values()))
        
        # Build result rows
        for combo in combinations:
            row = {main_col: a_value}
            row.update(dict(zip(other_cols, combo)))
            result_rows.append(row)
    
    return pd.DataFrame(result_rows)


def creat_graph_adata(file_path, meta_label, strong):
    with open(file_path + 'Graph_embedded.pickle', 'rb') as file:
        graph_dataset = pickle.load(file)
    graph_dataset = sorted(graph_dataset, key=lambda data: data.sample)
    adata = sc.read_h5ad(file_path + 'cell_embedded_adata.h5ad')
    adata.obs = adata.obs.reset_index(drop=True)
    adata.obs['cell_index'] = adata.obs.index

    if strong:
        graph_dataset_test = []
        for graph in graph_dataset:
            for i in range(30):
                sub_graph = sample_subgraph(graph, 1001)
                sub_graph.average_embedding = sub_graph.embedding.mean(dim=0)
                graph_dataset_test.append(sub_graph)
        graph_dataset_test, graph_dataset = graph_dataset, graph_dataset_test

    data_embedding = []
    for data in graph_dataset:
        embedding = data.average_embedding.numpy()
        data_embedding.append(embedding)
    
    meta_need = ["sample", "cluster", "sample_num", "cluster_num"] + meta_label
    meta = adata.obs[meta_need]
    meta_data_adata = explode_nested_relationships(adata.obs, "sample", meta_need)

    edge_num_csv = []
    train_test_csv = []
    sample_num_csv = []
    cell_num_csv = []
    for i in range(len(graph_dataset)):
        data = graph_dataset[i]
        edge_num = len(data.edge_index[0])/len(data.x)
        edge_num_csv.append(edge_num)
        if (data.x.size()[0])>1000:
            train_test_csv.append(data.train_test.numpy())
        else:
            train_test_csv.append(int(2))
        sample_num_csv.append(data.sample.numpy())
        cell_num_csv.append(len(data.x))
    meta_data_graph = pd.DataFrame({
        'sample_num': sample_num_csv,
        'averge_edge_num': edge_num_csv,
        'train/test': train_test_csv,
        'cell_num': cell_num_csv
    })

    meta_data_graph['sample_num'] = meta_data_graph['sample_num'].astype(int)
    meta_data = pd.merge(meta_data_adata, meta_data_graph, on='sample_num', how='outer')
    adata_emb = ad.AnnData(X=pd.DataFrame(data_embedding), obs=meta_data)
    for col in adata_emb.obs.columns:
        if adata_emb.obs[col].dtype == "object":
            adata_emb.obs[col] = adata_emb.obs[col].astype(str)
    return adata_emb



