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
        # x1: Original gene expression, x2: Perturbed gene expression
        if self.method == 'gaussian':
            x_twoview = add_gaussian_noise(x)
        elif self.method == 'mask':
            x_twoview = mask_genes(x)
        elif self.method == 'batch':
            x_twoview = de_batch(x)
        recon_x, _, mu, logvar, z = self.vae_view1(x)
        recon_x_twoview, _, mu_twoview, logvar_twoview, z_twoview = self.vae_view2(x_twoview)

        return x, recon_x, mu, logvar, z, x_twoview, recon_x_twoview, mu_twoview, logvar_twoview, z_twoview

# Loss function definition
def kl_annealing(epoch, total_epochs, max_beta):
    return min(max_beta, max_beta * epoch /20 )  

def loss_function_oneview(recon_x, x, mu, logvar, _, epoch=None, total_epochs=None):
    # Reconstruction loss
    reconstruction_loss = nn.MSELoss(reduction='mean')(recon_x, x)

    # KL divergence loss
    kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1))
    
    # KL annealing weight (default uses maximum value)
    max_beta=1e-5
    if epoch is not None and total_epochs is not None:
        kl_weight = kl_annealing(epoch, total_epochs, max_beta)
    else:
        kl_weight = 1e-5

    # Sparsity regularization (optional)
    beta = 1e-3
    sparsity_loss = beta * torch.sum(torch.abs(mu))  # Adjustable sparsity constraint weight
    return reconstruction_loss + kl_weight * kl_divergence + 0 * sparsity_loss, kl_weight, max_beta, reconstruction_loss, kl_weight * kl_divergence

def loss_function_twoview(x, recon_x, mu, logvar, z, 
                  x_twoview, recon_x_twoview, mu_twoview, logvar_twoview, z_twoview, epoch=None, total_epochs=None):
    # Reconstruction loss
    reconstruction_loss1 = nn.MSELoss()(recon_x, x)
    reconstruction_loss2 = nn.MSELoss()(recon_x_twoview, x_twoview)
    # KL divergence loss
    kl_divergence1 = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar), 1), 0)
    kl_divergence2 = torch.mean(-0.5 * torch.sum(1 + logvar_twoview - mu_twoview**2 - torch.exp(logvar_twoview), 1), 0)
    # KL annealing weight (default uses maximum value)
    max_beta=1
    if epoch is not None and total_epochs is not None:
        kl_weight = kl_annealing(epoch, total_epochs, max_beta)
    else:
        kl_weight = 1  # fallback (compatible with old calling method)
    # Sparsity constraint: L1 regularization term
    beta=1e-3
    sparsity_loss1 = beta * torch.sum(torch.abs(mu))  # Force mean of latent variables to approach zero
    sparsity_loss2 = beta * torch.sum(torch.abs(mu_twoview))  # Force mean of latent variables to approach zero
    # Contrastive loss (NCE)
    contrastive_loss = F.mse_loss(z, z_twoview)  # Can use more advanced contrastive loss, such as InfoNCE
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
        # ---------------------- 1. Synchronize before distributed training (key: ensure all processes train on the same epoch) ----------------------
        sampler_train.set_epoch(epoch)
        # Main process sends "whether to continue training" signal (initial is True)
        if dist.get_rank() == 0:
            continue_training = True  # Default is to continue training
        else:
            continue_training = False
        # Broadcast "whether to continue training" signal (if main process decides to stop, all processes exit loop directly)
        continue_tensor = torch.tensor(1 if continue_training else 0, device=device)
        dist.broadcast(continue_tensor, src=0)
        if continue_tensor.item() == 0:
            break  # Non-main processes receive "stop signal", exit directly

        # ---------------------- 2. Training phase (all processes execute, no difference) ----------------------
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

        # Synchronize training loss (all processes calculate global average loss)
        loss_tensor = torch.tensor(running_loss, device=device)
        count_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = (loss_tensor / count_tensor).item()

        # ---------------------- 3. Validation phase (all processes execute, synchronize test_loss) ----------------------
        test_loss = test_model(model, test_loader, loss_function, device, epoch, num_epochs)
        scheduler.step()

        # ---------------------- 4. Early stopping judgment (only main process executes, fully controls flow) ----------------------
        if dist.get_rank() == 0:
            # Print logs (only main process)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

            # Update best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                trigger_times = 0  # Reset counter
                # Save best weights
                if isinstance(model, DDP):
                    best_model_state = model.module.state_dict()
                else:
                    best_model_state = model.state_dict()
                print(f"Updated best model, current best validation loss: {best_test_loss:.4f}")
            else:
                # Only accumulate counter after KL annealing is complete
                if kl_weight >= max_beta:
                    trigger_times += 1
                    print(f"Early stopping counter: {trigger_times}/{patience}")
                    # Main process judges whether to early stop
                    if trigger_times >= patience:
                        print('Early stopping! Main process sends stop signal')
                        continue_training = False  # Main process sets to stop

        # ---------------------- 5. Synchronize before next training round (main process informs all processes whether to continue) ----------------------
        if dist.get_rank() == 0:
            continue_tensor = torch.tensor(1 if continue_training else 0, device=device)
        dist.broadcast(continue_tensor, src=0)
        if continue_tensor.item() == 0:
            break  # All processes receive "stop signal", exit loop

    # ---------------------- 6. Load best model (only main process) ----------------------
    if dist.get_rank() == 0 and best_model_state is not None:
        if isinstance(model, DDP):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
        print(f"Training ended, final best validation loss: {best_test_loss:.4f}")

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

    # Synchronize loss and sample count
    loss_tensor = torch.tensor(running_loss, device=device)
    count_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    return (loss_tensor / count_tensor).item()


# Use model to add embedding
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









