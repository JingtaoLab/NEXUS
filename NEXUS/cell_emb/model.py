import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.ln1 = nn.LayerNorm(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.dropout = nn.Dropout(0.1)
        self.init_parameters()

    def init_parameters(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_mu, self.fc_logvar]:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.gelu(self.ln3(self.fc3(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# ---------- Decoder ----------
class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.ln3 = nn.LayerNorm(2048)
        self.fc4 = nn.Linear(2048, input_dim)

        self.dropout = nn.Dropout(0.1)
        self.init_parameters()

    def init_parameters(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

    def forward(self, z):
        x = F.gelu(self.ln1(self.fc1(z)))
        x = self.dropout(x)
        x = F.gelu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.gelu(self.ln3(self.fc3(x)))
        x = self.fc4(x) 
        return x


# ---------- OneViewVAE  ----------
class OneViewVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(OneViewVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, x, mu, logvar, z

def add_gaussian_noise(x, noise_level=0.005):
    noise = torch.randn_like(x) * noise_level
    return x + noise

def mask_genes(x, mask_prob=0.01):
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask

def de_batch(x, de_batch=0.01):
    x = x - 0.01*torch.rand(1, device=x.device)
    return x

# ---------- TwoViewVAE  ----------
class TwoViewVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, method):
        super(TwoViewVAE, self).__init__()
        self.method = method
        self.vae_view1 = OneViewVAE(input_dim, latent_dim)
        self.vae_view2 = OneViewVAE(input_dim, latent_dim)

    def forward(self, x):
        # x1: 原始基因表达, x2: 扰动后的基因表达
        if self.method == 'gaussian':
            x_twoview = add_gaussian_noise(x)
        elif self.method == 'mask':
            x_twoview = mask_genes(x)
        elif self.method == 'batch':
            x_twoview = de_batch(x)
        recon_x, _, mu, logvar, z = self.vae_view1(x)
        recon_x_twoview, _, mu_twoview, logvar_twoview, z_twoview = self.vae_view2(x_twoview)

        return x, recon_x, mu, logvar, z, x_twoview, recon_x_twoview, mu_twoview, logvar_twoview, z_twoview

# 损失函数定义
def kl_annealing(epoch, total_epochs, max_beta):
    return min(max_beta, max_beta * epoch /20 )  

def loss_function_oneview(recon_x, x, mu, logvar, _, epoch=None, total_epochs=None):
    # 重构损失
    reconstruction_loss = nn.MSELoss(reduction='mean')(recon_x, x)

    # KL 散度损失
    kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1))
    
    # KL annealing 权重（默认使用最大值）
    max_beta=1e-5
    if epoch is not None and total_epochs is not None:
        kl_weight = kl_annealing(epoch, total_epochs, max_beta)
    else:
        kl_weight = 1e-5

    # 稀疏性正则（可选）
    beta = 1e-3
    sparsity_loss = beta * torch.sum(torch.abs(mu))  # 可调节稀疏性约束权重
    return reconstruction_loss + kl_weight * kl_divergence + 0 * sparsity_loss, kl_weight, max_beta, reconstruction_loss, kl_weight * kl_divergence

def loss_function_twoview(x, recon_x, mu, logvar, z, 
                  x_twoview, recon_x_twoview, mu_twoview, logvar_twoview, z_twoview, epoch=None, total_epochs=None):
    # 重构损失
    reconstruction_loss1 = nn.MSELoss()(recon_x, x)
    reconstruction_loss2 = nn.MSELoss()(recon_x_twoview, x_twoview)
    # KL 散度损失
    kl_divergence1 = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), 1), 0)
    kl_divergence2 = torch.mean(-0.5 * torch.sum(1 + logvar_twoview - mu_twoview**2 - torch.exp(logvar_twoview), 1), 0)
    # KL annealing 权重（默认使用最大值）
    max_beta=1
    if epoch is not None and total_epochs is not None:
        kl_weight = kl_annealing(epoch, total_epochs, max_beta)
    else:
        kl_weight = 1  # fallback（兼容旧调用方式）
    # 稀疏性约束：L1 正则化项
    beta=1e-3
    sparsity_loss1 = beta * torch.sum(torch.abs(mu))  # 强制潜在变量的均值接近零
    sparsity_loss2 = beta * torch.sum(torch.abs(mu_twoview))  # 强制潜在变量的均值接近零
    # 对比损失 (NCE)
    contrastive_loss = F.mse_loss(z, z_twoview)  # 可以使用更高级的对比损失，如InfoNCE
    loss = reconstruction_loss1 + reconstruction_loss2 + kl_weight*kl_divergence1 + kl_weight*kl_divergence2 + 0*sparsity_loss1 + 0*sparsity_loss1 + contrastive_loss

    return loss


def train_model(model, data_loader, test_loader, sampler_train, loss_function, device, num_epochs=500):
    best_test_loss = float('inf')
    best_model_state = None
    patience = 10
    trigger_times = 0
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    for epoch in range(num_epochs):
        # ---------------------- 1. 分布式训练前同步（关键：确保所有进程按同一epoch训练） ----------------------
        sampler_train.set_epoch(epoch)
        # 主进程发送“是否继续训练”的信号（初始为True）
        if dist.get_rank() == 0:
            continue_training = True  # 默认为继续训练
        else:
            continue_training = False
        # 广播“是否继续训练”的信号（若主进程决定停止，所有进程直接退出循环）
        continue_tensor = torch.tensor(1 if continue_training else 0, device=device)
        dist.broadcast(continue_tensor, src=0)
        if continue_tensor.item() == 0:
            break  # 非主进程接收“停止信号”，直接退出

        # ---------------------- 2. 训练阶段（所有进程执行，无差异） ----------------------
        running_loss = 0.0
        total_samples = 0
        for data in data_loader:
            inputs = data.to(device)
            optimizer.zero_grad()
            loss_output = loss_function(*model(inputs), epoch=epoch, total_epochs=num_epochs)
            loss = loss_output[0]
            kl_weight = loss_output[1]
            max_beta = loss_output[2]
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        # 同步训练损失（所有进程计算全局平均损失）
        loss_tensor = torch.tensor(running_loss, device=device)
        count_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = (loss_tensor / count_tensor).item()

        # ---------------------- 3. 验证阶段（所有进程执行，同步test_loss） ----------------------
        test_loss = test_model(model, test_loader, loss_function, device, epoch, num_epochs)
        scheduler.step()

        # ---------------------- 4. 早停判断（仅主进程执行，完全控制流程） ----------------------
        if dist.get_rank() == 0:
            # 打印日志（仅主进程）
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

            # 更新最佳模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                trigger_times = 0  # 重置计数器
                # 保存最佳权重
                if isinstance(model, DDP):
                    best_model_state = model.module.state_dict()
                else:
                    best_model_state = model.state_dict()
                print(f"更新最佳模型，当前最佳验证损失: {best_test_loss:.4f}")
            else:
                # 仅在KL退火完成后累加计数器
                if kl_weight >= max_beta:
                    trigger_times += 1
                    print(f"早停计数器: {trigger_times}/{patience}")
                    # 主进程判断是否早停
                    if trigger_times >= patience:
                        print('Early stopping! 主进程发送停止信号')
                        continue_training = False  # 主进程设置为停止

        # ---------------------- 5. 下一轮训练前同步（主进程告知所有进程是否继续） ----------------------
        if dist.get_rank() == 0:
            continue_tensor = torch.tensor(1 if continue_training else 0, device=device)
        dist.broadcast(continue_tensor, src=0)
        if continue_tensor.item() == 0:
            break  # 所有进程接收“停止信号”，退出循环

    # ---------------------- 6. 加载最佳模型（仅主进程） ----------------------
    if dist.get_rank() == 0 and best_model_state is not None:
        if isinstance(model, DDP):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
        print(f"训练结束，最终最佳验证损失: {best_test_loss:.4f}")

    return model

def test_model(model, data_loader, loss_function, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in data_loader:
            inputs = data.to(device)
            loss,_,_,_,_ = loss_function(*model(inputs), epoch=epoch, total_epochs=num_epochs)
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    # 同步 loss 和样本数
    loss_tensor = torch.tensor(running_loss, device=device)
    count_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    return (loss_tensor / count_tensor).item()


# 使用模型添加embedding
def embedding(model, data_emb, adata, view):
    model.eval()
    with torch.no_grad():
        counts_raw = []
        counts_recon = []
        embeddings = []
        for data in data_emb:
            inputs = data
            counts_raw.append(inputs)
            if view == "oneview":
                recon_x, _, _, _, z = model(inputs)
            elif view == "twoview":
                recon_x, _, _, _, z = model.vae_view1(inputs)
            counts_recon.append(recon_x)
            embeddings.append(z)
        counts_raw = torch.cat(counts_raw, 0)
        counts_recon = torch.cat(counts_recon, 0)
        embeddings = torch.cat(embeddings, 0)
        embeddings = embeddings.cpu().numpy().astype('float64')
        
        adata_raw = adata
        adata_recon = adata
        adata_raw.X = counts_raw
        adata_raw.obsm['X_VAE'] = embeddings
        adata_recon.X = counts_recon
        adata_recon.obsm['X_VAE'] = embeddings

        return adata_raw, adata_recon


class EncoderOnly(nn.Module): 
    def __init__(self, original_model, view):
        super(EncoderOnly, self).__init__()
        self.view = view
        if self.view == "oneview":
            self.encoder = original_model.encoder
        if self.view == "twoview":
            self.encoder = original_model.vae_view1.encoder
        
    def forward(self, x):
        mu, _ = self.encoder(x)
        return mu









