import torch
import torch.nn as nn

from modules import (
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    CausalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
)


class CDRL4AD(nn.Module):

    def __init__(
        self,
        n_features,
        topk,
        window_size,
        cause_window_size,
        causal_hid_dim,
        out_dim,
        device,
        embed_dim=64,
        causal_thres=0.1,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.3,
        alpha=0.2,
    ):
        super(CDRL4AD, self).__init__()

        self.n_features = n_features
        self.topk = topk
        self.feature_gat = FeatureAttentionLayer(n_features, topk, window_size, dropout, alpha, embed_dim)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, embed_dim)
        self.causal_gat = CausalAttentionLayer(n_features, cause_window_size, dropout, alpha, causal_thres, causal_hid_dim, device)
        self.feat_lin = nn.Linear(topk, 1)
        self.gru = GRULayer(2 * window_size + embed_dim + causal_hid_dim, gru_hid_dim, gru_n_layers, dropout) # (4 * window_size + topk + 1)
        self.forecasting_model = Forecasting_Model(n_features, gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(n_features, window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)
        self.fc = nn.Linear(n_features, n_features)
        self.embedding = nn.Embedding(n_features, embed_dim)
        self.device = device

    def forward(self, x, y): 
        all_embeddings = self.embedding(torch.arange(self.n_features).to(self.device))

        weights_arr = all_embeddings.detach().clone()
        all_embeddings = all_embeddings.repeat(16, 1)

        weights = weights_arr.view(self.n_features, -1)

        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
        cos_ji_mat = cos_ji_mat / normed_mat
        cos_ji_mat.fill_diagonal_(0.0)

        dim = weights.shape[-1]

        edge_features, topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)

        gated_i = torch.arange(0, self.n_features).T.unsqueeze(1).repeat(1, self.topk).flatten().to(self.device).unsqueeze(0)
        gated_j = topk_indices_ji.flatten().unsqueeze(0)

        edge_indices = torch.cat((gated_j, gated_i), dim=0)    
        edge_features = edge_features.flatten().unsqueeze(0)

        h_feat = self.feature_gat(x, edge_features, edge_indices, all_embeddings)
        h_feat = self.feat_lin(h_feat.permute(0, 1, 3, 2)).squeeze(-1)
        h_temp = self.temporal_gat(x, edge_features)
        h_cause = self.causal_gat(x, y) 

        h_cat = torch.cat([x, h_feat, h_temp, h_cause], dim=2)  

        _, h_end = self.gru(h_cat)              
        h_end = h_end.view(x.shape[0], -1) 

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons
