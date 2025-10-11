import torch
import torch.nn as nn
import torch.nn.init as init

class FeatureAttentionLayer(nn.Module):

    def __init__(self, n_features, topk, window_size, dropout, alpha, embed_dim=None, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.topk = topk
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.num_nodes = n_features
        self.use_bias = use_bias

        a_input_dim = 2 * self.embed_dim  

        self.lin_n = nn.Linear(window_size, self.embed_dim)      
        self.lin_e = nn.Linear(1, self.embed_dim)         
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        self.fc = nn.Linear(topk*n_features, n_features)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias_n = nn.Parameter(torch.empty(n_features, topk))
            self.bias_e = nn.Parameter(torch.empty(n_features, topk))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x_n, x_e, edge_indices, all_embeddings):
        Wx_n = self.lin_n(x_n)                                                     

        a_input_n = self._make_node_attention_input(Wx_n, edge_indices)            

        x_e = x_e.repeat(x_n.shape[0], 1).view(x_n.shape[0], self.n_features, self.topk, 1)
        Wx_e = self.lin_e(x_e)                                              
        a_input_e = self._make_edge_attention_input(Wx_n, Wx_e, edge_indices)            

        e_n = self.leakyrelu(torch.matmul(a_input_n, self.a)).squeeze(3)     
        e_e = self.leakyrelu(torch.matmul(a_input_e, self.a)).squeeze(3)     

        if self.use_bias:
            e_n += torch.nan_to_num(self.bias_n)
            e_e += torch.nan_to_num(self.bias_e)

        attention_n = torch.softmax(e_n, dim=2)
        attention_n = torch.dropout(attention_n, self.dropout, train=self.training)  
        attention_e = torch.softmax(e_e, dim=2)
        attention_e = torch.dropout(attention_e, self.dropout, train=self.training)  

        h_n = self.leakyrelu(torch.mul(attention_n.unsqueeze(-1), Wx_n[:, edge_indices[0].reshape(self.n_features, self.topk)]).view(x_n.shape[0], self.n_features, self.topk, self.embed_dim))
        h_e = self.leakyrelu(torch.mul(attention_e, x_e.squeeze(3)).view(x_e.shape[0],self.n_features, self.topk))

        h_e = self.lin_e(h_e.unsqueeze(-1))

        h_cat = torch.cat([h_n, h_e], dim=2)  
        h_cat = self.leakyrelu(h_cat)

        return h_n 

    def _make_node_attention_input(self, v, edge_indices):
        K = self.topk
        blocks_repeating = v.repeat_interleave(K, dim=1).view(v.shape[0], self.n_features, K, self.embed_dim)  
        blocks_alternating = v[:, edge_indices[0].reshape(self.n_features, self.topk)]
        
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        return combined.view(v.size(0), self.n_features, K, 2 * self.embed_dim)


    def _make_edge_attention_input(self, v_n, v_e, edge_indices):
        K = self.n_features

        blocks_repeating = v_n.repeat_interleave(self.topk, dim=1).view(v_n.shape[0], self.n_features, self.topk, self.embed_dim)
        blocks_alternating = v_e               

        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)
        return combined.view(v_e.size(0), self.num_nodes, self.topk, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        lin_input_dim = n_features
        a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()

    def forward(self, x, edge_features):
        x = x.permute(0, 2, 1)  

        Wx = self.lin(x)                                                  
        a_input = self._make_attention_input(Wx)                         
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)   

        if self.use_bias:
            e += torch.nan_to_num(self.bias) 

        attention = torch.softmax(e, dim=2)  
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.relu(torch.matmul(attention, x)) 
        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        return combined.view(v.size(0), K, K, 2 * self.embed_dim)

class CausalAttentionLayer(nn.Module):

    def __init__(self, n_features, window_size, dropout, alpha, causal_thres, hid_dim, device, use_bias=True):
        super(CausalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = window_size 
        self.causal_thres = causal_thres
        self.num_nodes = n_features
        self.use_bias = use_bias
        self.hidden = hid_dim
        self.device = device

        a_input_dim = 2

        self.lin = nn.Linear(window_size, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_features, window_size*n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)

        self.gru_left = nn.GRU(window_size, self.hidden, batch_first=True)
        self.gru_left.flatten_parameters()

        self.fc_mu = nn.Linear(self.hidden, self.hidden)
        self.fc_std = nn.Linear(self.hidden, self.hidden)

        self.networks = nn.ModuleList([
            GRU(window_size, self.hidden) for i in range(n_features)])

    def forward(self, x, y):

        x = x[:, :, -self.window_size:] 

        Wx = self.lin(x)                                      
        a_input = self._make_attention_input(Wx, y)                        

        e = torch.matmul(a_input, self.a).squeeze(3)
        if self.use_bias:
            e += self.bias
        e = self.leakyrelu(e)

        attention = torch.softmax(e, dim=2)  
        attention = torch.dropout(attention, self.dropout, train=self.training)

        A = (attention >= self.causal_thres).int()
        Ax = A * a_input[:, :, :, 1].squeeze(2)
        cause_x = Wx + Ax.view(x.shape[0], self.n_features,  -1).sum(dim=2).unsqueeze(-1)

        hidden_0 = torch.zeros(1, x.shape[0], self.hidden).to(self.device)
        out, h_t = self.gru_left(cause_x, hidden_0)

        mu = self.fc_mu(h_t)
        log_var = self.fc_std(h_t)

        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = mu.size())
        z = z.type_as(mu)
        z = mu + sigma*z

        causes = [self.networks[i](Wx, z)[1] for i in range(self.n_features)]
        cause_x = torch.cat(causes, dim=0)

        cause_x = cause_x.permute(1, 0, 2)

        return cause_x 

    def _make_attention_input(self, v, y):
        K = self.num_nodes
        v = v.view(v.shape[0], -1)

        blocks_repeating = y.unsqueeze(-1).repeat_interleave(K * self.window_size, dim=2)
        blocks_alternating = v.repeat(1, 1, K).view(v.shape[0], K, K*self.window_size).unsqueeze(-1)  
        
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=3)

        return combined.view(v.size(0), K, K * self.window_size, -1)

class GRU(nn.Module):
    def __init__(self, num_series, hidden):

        super(GRU, self).__init__()
        self.p = num_series
        self.hidden = hidden

        # Set up network.
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.linear = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch):
        return torch.zeros(1, batch, self.hidden)

    def forward(self, X, z, mode = 'train'):
        if mode == 'train':
          X_right, hidden_out = self.gru(X, z)
          X_right = self.linear(X_right)

          return X_right, hidden_out

class GRULayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  
        return out, h


class RNNDecoder(nn.Module):

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out

class ReconstructionModel(nn.Module):

    def __init__(self, n_features, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, window_size)

    def forward(self, x):
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.n_features, dim=1).view(x.size(0), self.n_features, -1)

        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    def __init__(self, n_features, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.n_features = n_features
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.leakyRelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.leakyRelu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x).view(x.shape[0], self.n_features)
