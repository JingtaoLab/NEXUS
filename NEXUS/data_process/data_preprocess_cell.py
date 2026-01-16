import gc
import scanpy as sc
import numpy as np
import harmonypy as harmony
import scipy.sparse as sp
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class Data_process:
    def __init__(self, input_file_path, num_of_highly_variable_genes, split_ratio, batch_size, device_use, process=True):
        self.input_file_path = input_file_path
        self.num_of_highly_variable_genes = num_of_highly_variable_genes
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.device_use = device_use
        self.process = process
        self.use_ddp = False  # Flag indicating whether to use DDP
        self.world_size = 1  # Number of processes

    def read_all_adata(self):
        # Read combined_adata
        if self.input_file_path.endswith(".h5ad"):
            # Read h5ad file (if it's an integrated file, read h5ad directly, needs to be raw counts)
            self.combined_adata = sc.read(self.input_file_path)
            gene_names = self.combined_adata.var_names
            self.combined_adata = self.combined_adata[:, ~gene_names.duplicated(keep='first')]
        else:
            # Read h5ad files (if not integrated, read from folder and integrate, needs to be raw counts)
            scRNA_datas = []
            for file_name in os.listdir(self.input_file_path):
                if file_name.endswith(".h5ad"):
                    adata=sc.read_h5ad(self.input_file_path+file_name)
                    sample = file_name.split(".")
                    sample = sample[0]
                    adata.obs['sample'] = sample
                    adata.var.index = adata.var.index.astype(str)
                    # Remove duplicate genes
                    gene_names = adata.var_names
                    adata = adata[:, ~gene_names.duplicated(keep='first')]
                    scRNA_datas.append(adata)
            self.combined_adata = scRNA_datas[0].concatenate(*scRNA_datas[1:], join='outer')
            self.combined_adata = self.combined_adata
            del scRNA_datas
            gc.collect()

        if self.process:
            self.combined_adata = self.combined_adata[self.combined_adata.X.sum(axis=1) > 0]    # Filter empty cells
            sc.pp.filter_cells(self.combined_adata, min_genes=3)
            sc.pp.filter_genes(self.combined_adata, min_cells=200)
            self.combined_adata.var = self.combined_adata.var.astype(str)
            self.combined_adata.obs = self.combined_adata.obs.astype(str)
            self.combined_adata.raw = self.combined_adata
            sc.pp.normalize_total(self.combined_adata, target_sum=1e4)
            sc.pp.log1p(self.combined_adata)
            sc.pp.highly_variable_genes(self.combined_adata, n_top_genes = self.num_of_highly_variable_genes)
            self.combined_adata.var = self.combined_adata.var.copy()
            self.combined_adata = self.combined_adata[:, self.combined_adata.var['highly_variable']].copy()
            sc.pp.scale(self.combined_adata, max_value=10)

    def harmony(self):
        self.X = self.combined_adata.X.toarray() if isinstance(self.combined_adata.X, np.ndarray) == False else self.combined_adata.X
        self.X_harmony = harmony.run_harmony(self.X, self.combined_adata.obs, ['sample'])
        self.combined_adata.X = self.X_harmony.Z_corr.T
        self.combined_adata.var = self.combined_adata.var.astype(str)

    def train_test_split(self):
        # Get cell indices
        cell_indices = np.arange(self.combined_adata.n_obs)
        # Randomly shuffle cell indices
        np.random.shuffle(cell_indices)
        # Split indices
        split_idx = int(self.split_ratio * len(cell_indices))
        train_indices = cell_indices[:split_idx]
        test_indices = cell_indices[split_idx:]
        # Create training and test sets
        self.adata_train = self.combined_adata[train_indices].copy()
        self.adata_test = self.combined_adata[test_indices].copy()

        return self.adata_train, self.adata_test

    def setup_DDP(self):
        if self.device_use == "gpu":
            # Detect available GPU count
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Please check your GPU setup.")
            
            num_gpus = torch.cuda.device_count()
            print(f"Detected {num_gpus} GPU(s)")
            
            # Check if environment variables are set (may be set by external startup script like torchrun)
            rank = os.environ.get('RANK', None)
            world_size = os.environ.get('WORLD_SIZE', None)
            
            if rank is not None and world_size is not None:
                # Environment variables are set, use multi-process startup (e.g., torchrun)
                rank = int(rank)
                world_size = int(world_size)
                local_rank = int(os.environ.get('LOCAL_RANK', rank))
                
                if not dist.is_initialized():
                    dist.init_process_group(backend='nccl', init_method='env://')
                torch.cuda.set_device(local_rank)
                self.device = torch.device("cuda", local_rank)
                self.use_ddp = True
                self.world_size = world_size
                print(f"Using DDP with {world_size} processes. Rank: {rank}, Local Rank: {local_rank}")
            elif num_gpus > 1:
                # Multiple GPUs but environment variables not set, can choose to automatically set environment variables to use multi-GPU DDP
                # or use single GPU mode
                # Here we automatically set environment variables to use all available GPUs
                print(f"Multiple GPUs detected ({num_gpus} GPUs) but RANK/WORLD_SIZE not set.")
                print(f"Auto-configuring DDP to use all {num_gpus} GPUs...")
                
                # Automatically set environment variables to use all GPUs (single-process multi-GPU mode)
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = str(num_gpus)
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                
                # Note: Single-process multi-GPU DDP requires special handling, here we use single GPU mode
                # True multi-GPU DDP should be started with torchrun
                print(f"Note: For true multi-GPU DDP, please use: torchrun --nproc_per_node={num_gpus} your_script.py")
                print(f"Using single GPU (GPU 0) for now. To use all GPUs, set environment variables or use torchrun.")
                self.device = torch.device("cuda", 0)
                self.use_ddp = False
                self.world_size = 1
            else:
                # Only one GPU, use single GPU mode
                print("Single GPU detected. Using single GPU mode (no DDP).")
                self.device = torch.device("cuda", 0)
                self.use_ddp = False
                self.world_size = 1
    
        elif self.device_use == "cpu":
            # CPU mode: check if already initialized
            if not dist.is_initialized():
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = '1'
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                dist.init_process_group(backend='gloo', init_method='env://')
            rank = dist.get_rank()
            self.device = torch.device("cpu")
            self.use_ddp = True  # Also use DDP structure in CPU mode
            self.world_size = 1
        
        else:
            raise ValueError("Invalid device_use. Must be 'gpu' or 'cpu'.")

        return self.device
                
    
    def data_sampler(self, train, test):
        if sp.issparse(train.X):
            train_X = torch.tensor(train.X.toarray()).float()
        else:
            train_X = torch.tensor(train.X).float()

        if sp.issparse(test.X):
            test_X = torch.tensor(test.X.toarray()).float()
        else:
            test_X = torch.tensor(test.X).float()

        datasets_train = {}
        for i in range(len(train_X)):
            datasets_train[i] = (train_X[i])

        datasets_test = {}
        for i in range(len(test_X)):
            datasets_test[i] = (test_X[i])

        batch_size = self.batch_size
        
        # Choose different sampler based on whether DDP is used
        if self.use_ddp:
            # Use DistributedSampler in DDP mode
            sampler_train = DistributedSampler(datasets_train, shuffle=True)
            sampler_test = DistributedSampler(datasets_test, shuffle=True)
            train_iter = DataLoader(datasets_train, batch_size, shuffle=False, drop_last=False, sampler=sampler_train)
            test_iter = DataLoader(datasets_test, batch_size, shuffle=False, drop_last=False, sampler=sampler_test)
        else:
            # Single GPU mode, do not use DistributedSampler
            sampler_train = None
            sampler_test = None
            train_iter = DataLoader(datasets_train, batch_size, shuffle=True, drop_last=False)
            test_iter = DataLoader(datasets_test, batch_size, shuffle=True, drop_last=False)
        
        train_iter_emb = DataLoader(datasets_train, batch_size, shuffle=False, drop_last=False)
        test_iter_emb = DataLoader(datasets_test, batch_size, shuffle=False, drop_last=False)

        
        return train, test, train_iter, test_iter, train_iter_emb, test_iter_emb, sampler_train, sampler_test
        
