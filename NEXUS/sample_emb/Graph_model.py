import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGPooling, GCNConv, TransformerConv
from torch_geometric.nn import GraphConv, TopKPooling, global_mean_pool, GlobalAttention, global_max_pool
from torch.cuda.amp import autocast, GradScaler

# Parameters to be defined
#input_dim
#hidden_dim
#latent_dim
#class_num

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, heads, edge_dims):
        super(GraphEncoder, self).__init__()
        self.hidden_size = hidden_dim*heads
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads, edge_dim=edge_dims)
        self.bn1 = nn.LayerNorm(self.hidden_size)
        self.conv2 = TransformerConv(self.hidden_size, hidden_dim, heads, edge_dim=edge_dims)
        self.bn2 = nn.LayerNorm(self.hidden_size)
        self.conv3 = TransformerConv(self.hidden_size, latent_dim//heads, heads, edge_dim=edge_dims)
        self.dropout = torch.nn.Dropout(p=0.1)
        
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x, edge_index, attr):
        x, attn1 = self._block(self.conv1, self.bn1, x, edge_index, attr)
        x, attn2 = self._block(self.conv2, self.bn2, x, edge_index, attr)
        x, attn3 = self.conv3(x, edge_index, attr, return_attention_weights = True)
        return x, attn1, attn2, attn3

    def _block(self, conv, norm, x, edge_index, attr):
        x, attn = conv(x, edge_index, attr, return_attention_weights = True)
        x = norm(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return x, attn



# Define classifier model
class Classifier(nn.Module):
    def __init__(self, latent_dim, class_num):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.bn1 = nn.LayerNorm(latent_dim)
        self.fc2 = nn.Linear(latent_dim, class_num)
        self.dropout = torch.nn.Dropout(p=0.1)
        
        self.init_parameters()
        
    def init_parameters(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)
        init.ones_(self.bn1.weight)
        init.zeros_(self.bn1.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, class_num, pooling_type='mean', heads = 4, edge_dims = 1):
        super(GraphVAE, self).__init__()
        self.pooling_type = pooling_type
        self.encoder = GraphEncoder(input_dim, hidden_dim, latent_dim, heads, edge_dims)
        self.class_num = class_num
        self.Classifier = Classifier(latent_dim, class_num)
        

    def forward(self, x, edge_index, attr, batch):
        z, attn1, attn2, attn3 = self.encoder(x, edge_index, attr)
            # Pooling layer
        if self.pooling_type == 'mean':
            z_pool = global_mean_pool(z, batch)
        elif self.pooling_type == 'max':
            z_pool = global_max_pool(z, batch)
        elif self.pooling_type == 'topk':
            z_pool = TopKPooling(z, edge_index, attr, batch)
        elif self.pooling_type == 'attention':
            z_pool = GlobalAttention(torch.nn.Sequential(torch.nn.Linear(latent_dim, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)))
        else:
            raise ValueError("Unknown pooling type")
        out_put = self.Classifier(z_pool)
        out_put = F.softmax(out_put, dim=1)
        z_pool = z_pool[batch[0]]
        return z, z_pool, out_put, attn1, attn2, attn3

# Train model
def train_model(model, data_loader, test_loader, loss_function, fine_tune, num_epochs=500, device='cpu'):
    sub_num = 1000
    last_test_loss = float('inf')
    patience = 5
    trigger_times = 0

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for data in data_loader:
            optimizer.zero_grad()
            batch = data.batch

            graphs, labels_graph = [], []
            for i in range(int(batch.max()) + 1):
                mask = batch == i
                graph = data.subgraph(mask)
                graphs.append(graph)
                labels_graph.append(graph.cluster)

            sub_graph_a, sub_graph_b = [], []
            for graph in graphs:
                sub_graph_a.append(sample_subgraph(graph, sub_num))
                sub_graph_b.append(sample_subgraph(graph, sub_num))

            z_a_list, z_b_list = [], []
            for sub_a, sub_b in zip(sub_graph_a, sub_graph_b):
                sub_a, sub_b = sub_a.to(device), sub_b.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    _, a_pool, *_ = model(sub_a.x, sub_a.edge_index, sub_a.edge_attr, sub_a.batch)
                    _, b_pool, *_ = model(sub_b.x, sub_b.edge_index, sub_b.edge_attr, sub_b.batch)
                z_a_list.append(a_pool)
                z_b_list.append(b_pool)

            z_a = torch.cat(z_a_list, dim=0).view(len(z_a_list), -1)
            z_b = torch.cat(z_b_list, dim=0).view(len(z_b_list), -1)

            data = sample_subgraph(data, sub_num)
            inputs, edge_index = data.x.to(device), data.edge_index.to(device)
            cluster, attr, batch = data.cluster.to(device), data.edge_attr.to(device), data.batch.to(device)

            with autocast(enabled=(device.type == 'cuda')):
                z_full, _, pre, *_ = model(inputs, edge_index, attr, batch)
                for p, a in zip(pre, cluster):
                    if torch.argmax(p) == a:
                        correct += 1
                loss, contras_loss, edge_pred_loss = loss_function(z_a, z_b, pre, cluster, z_full, edge_index, fine_tune)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()*len(cluster)
            total += len(cluster)

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Contrastive Loss: {contras_loss:.4f}, Edge Prediction Loss: {edge_pred_loss:.4f}")

        test_loss, test_acc = test_model(model, test_loader, device, sub_num, fine_tune)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        scheduler.step()

        if test_loss < last_test_loss:
            if abs(1 - test_loss / last_test_loss) > 0.0001:
                last_test_loss = test_loss
                trigger_times = 0
            else:
                trigger_times += 1
        else:
            trigger_times += 1

        if trigger_times >= patience:
            print("Early stopping!")
            break

        torch.cuda.empty_cache()
        
# Test model
def test_model(model, data_loader, device, sub_num, fine_tune, num_repeats=5):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            batch = data.batch
            graphs, labels_graph = [], []
            for i in range(int(batch.max()) + 1):
                mask = batch == i
                graph = data.subgraph(mask)
                labels_graph.append(graph.cluster)
                graphs.append(graph)

            x = len(labels_graph)
            z_a_list, z_b_list = [], []
            for _ in range(num_repeats):
                sub_graph_a, sub_graph_b = [], []
                for graph in graphs:
                    sub_graph_a.append(sample_subgraph(graph, sub_num))
                    sub_graph_b.append(sample_subgraph(graph, sub_num))

                z_a_repeat, z_b_repeat = [], []
                for sub_a, sub_b in zip(sub_graph_a, sub_graph_b):
                    sub_a = sub_a.to(device)
                    sub_b = sub_b.to(device)
                    with autocast(enabled=(device.type == 'cuda')):
                        _, a_pool, *_ = model(sub_a.x, sub_a.edge_index, sub_a.edge_attr, sub_a.batch)
                        _, b_pool, *_ = model(sub_b.x, sub_b.edge_index, sub_b.edge_attr, sub_b.batch)
                    z_a_repeat.append(a_pool)
                    z_b_repeat.append(b_pool)

                z_a_list.append(torch.stack(z_a_repeat))
                z_b_list.append(torch.stack(z_b_repeat))

            z_a = torch.stack(z_a_list).mean(dim=0)
            z_b = torch.stack(z_b_list).mean(dim=0)

            data = data.to(device)
            inputs, edge_index, attr, cluster, batch = data.x, data.edge_index, data.edge_attr, data.cluster, data.batch

            with autocast(enabled=(device.type == 'cuda')):
                z_full, _, pre, *_ = model(inputs, edge_index, attr, batch)

                for p, a in zip(pre, cluster):
                    if torch.argmax(p) == a:
                        correct += 1
                loss, _, _ = loss_function(z_a, z_b, pre, cluster, z_full, edge_index, fine_tune)

            running_loss += loss.item()*len(cluster)
            total += len(cluster)

    test_loss = running_loss / total
    test_acc = correct / total
    return test_loss, test_acc


# Use model to add embedding
def embedding(model, dataset, device):
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            inputs, edge_index, attr = dataset[i].x.to(device), dataset[i].edge_index.to(device), dataset[i].edge_attr.to(device)
            batch = torch.zeros(dataset[i].x.size()[0])
            batch = batch.to(torch.int64).to(device)
            z, z_pool, _, (_,attn1), (_,attn2), (_,attn3) = model(inputs, edge_index, attr, batch)
            dataset[i].embedding = z.to('cpu')
            dataset[i].average_embedding = z_pool.to('cpu')
            dataset[i].attn1 = attn1.to('cpu')
            dataset[i].attn2 = attn2.to('cpu')
            dataset[i].attn3 = attn3.to('cpu')
        return dataset
        print(f"Embedding done!")


def contrastive_loss(z_i, z_j, temperature=0.1):
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
    sim = torch.mm(z, z.t()) / temperature  # [2N, 2N]

    # Mask self-comparisons
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i + N)
    pos = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    sim_pos = sim[torch.arange(2 * N), pos]

    loss = -sim_pos + torch.logsumexp(sim, dim=1)
    return loss.mean()

def edge_prediction_loss(z, edge_index, num_neg_samples=1):
    num_nodes = z.size(0)
    pos_edge_index = edge_index.t()
    pos_scores = (z[pos_edge_index[:, 0]] * z[pos_edge_index[:, 1]]).sum(dim=1)
    neg_edge_index = torch.randint(0, num_nodes, (pos_edge_index.size(0) * num_neg_samples, 2), device=z.device)
    neg_scores = (z[neg_edge_index[:, 0]] * z[neg_edge_index[:, 1]]).sum(dim=1)
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
    return F.binary_cross_entropy_with_logits(scores, labels)

def loss_function(z_i, z_j, pre, cluster, z_full, edge_index, fine_tune):
    contras_loss = contrastive_loss(z_i, z_j)
    pre_loss = nn.NLLLoss()(pre, cluster)
    edge_pred_loss = edge_prediction_loss(z_full, edge_index)
    
    alpha = 0

    if fine_tune:
        loss_final = pre_loss
    else:
        loss_final = contras_loss + alpha * edge_pred_loss

    return loss_final, contras_loss, edge_pred_loss


def sample_subgraph(data, num_nodes):
    """
    Use `subgraph` method to randomly select subgraph from graph and preserve edge features.
    """
    # Get number of nodes in original graph
    total_nodes = data.x.size(0)

    if total_nodes > num_nodes:
        # Randomly select indices of `num_nodes` nodes
        selected_nodes = random.sample(range(total_nodes), num_nodes)
        selected_nodes = torch.tensor(selected_nodes, dtype=torch.long)

        # Use subgraph method to extract edge connection information (edge_index) and edge index mapping (e_id) of subgraph
        subgraph = data.subgraph(selected_nodes)  # Get edge connection information of subgraph
    else:
        subgraph = data

    return subgraph

