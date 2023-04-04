import torch
from torch import nn

from torch.nn.utils.weight_norm import WeightNorm


class TransformerModel(nn.Module):
    def __init__(self, n_way: int = 6, n_feature: int = 64,
                 n_head: int = 4, n_encoder_layers: int = 5,
                 dim_projection: int = 32, dim_feedforward: int = 32,
                 loss_type: str = 'dist'):
        super(TransformerModel, self).__init__()

        self.n_feature = n_feature
        self.n_head = n_head
        self.dim_projection = dim_projection
        self.d_model = dim_projection
        self.loss_type = loss_type  # 'softmax' #'dist'

        # Patch encoder
        self.patch_encoder = nn.Linear(self.n_feature, self.dim_projection)
        self.pos_encoder = nn.Parameter(torch.randn(64, self.dim_projection) * 1e-1)

        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head,
                                                   dim_feedforward=dim_feedforward)
        encoder_norm = nn.LayerNorm(self.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers, encoder_norm)

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.d_model, n_way)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':
            self.classifier = distLinear(self.d_model, n_way)

    def forward(self, src):
        src = src / torch.max(src)
        # src shape: [n_time, setsz, n_feature]
        # linearly transform a patch by projecting it into a vector of size 'dim_projection'
        batch_sz = src.shape[1]
        src = self.patch_encoder(src)
        src += torch.unsqueeze(self.pos_encoder, 1).repeat(1, batch_sz, 1)
        # INPUT SIZE: (n_steps, batch_size, n_features)
        memory = self.encoder(src)

        # Average the time dimention
        memory = memory[-1]
        # classify the features
        scores = self.classifier(memory)

        return scores


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)

        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)

        return scores

