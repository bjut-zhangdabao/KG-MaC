import torch
import torch.nn as nn
import torch.nn.functional as F

from src.CrossmodalTransformer import MULTModel
from src.StoG import CapsuleSequenceToGraph
from src.layer import TuckERLayer, MutanLayer
from BioCycleModel.Multimodal_Model import Text_Noise_Pre, Audio_Noise_Pre, Visual_Noise_Pre

import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz

from manifolds import Lorentz
from modules.hyper_nets import LorentzPositionwiseFeedForward, LorentzLinear
from modules.lmulti_headed_attn import LorentzMultiHeadedAttention
import math
from lib.lorentz.layers import LorentzConv1d
from lib.lorentz.layers import LorentzMLR

class LorentzFullyConnected(nn.Module):

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False
        ):
        super(LorentzFullyConnected, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):

        x = self.weight(x)
        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            mask = mask==False
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            num_features: int,
            num_classes: int
    ):
        super(LorentzMLR, self).__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(torch.zeros(num_classes, ))
        self.z = torch.nn.Parameter(
            F.pad(torch.zeros(num_classes, num_features - 2), pad=(1, 0), value=1))  # z should not be (0,0)

        self.init_weights()

    def forward(self, x):
        # Hyperplane
        sqrt_mK = 1 / self.manifold.k.sqrt()
        norm_z = torch.norm(self.z, dim=-1)
        w_t = (torch.sinh(sqrt_mK * self.a) * norm_z)
        w_s = torch.cosh(sqrt_mK * self.a.view(-1, 1)) * self.z
        beta = torch.sqrt(-w_t ** 2 + torch.norm(w_s, dim=-1) ** 2)
        alpha = -w_t * x.narrow(-1, 0, 1) + (torch.cosh(sqrt_mK * self.a) * torch.inner(x.narrow(-1, 1, x.shape[-1] - 1), self.z))

        d = self.manifold.k.sqrt() * torch.abs(torch.asinh(sqrt_mK * alpha / beta))  # Distance to hyperplane
        logits = torch.sign(alpha) * beta * d

        return logits

    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))
        return x

class LorentzTransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model=256, heads=8, d_ff=None, dropout=0.1, attention_dropout=0.1,
                 max_relative_positions=0):
        super(LorentzTransformerEncoderLayer, self).__init__()

        if d_ff is None:
            d_ff = d_model * 4
        self.manifold = Lorentz()
        self.self_attn = LorentzMultiHeadedAttention(
            heads, d_model, self.manifold, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = LorentzPositionwiseFeedForward(d_model, d_ff, self.manifold, dropout=dropout)
        self.dropout = nn.Dropout(0.1)
        self.residual = LorentzLinear(d_model, d_model, dropout=dropout, head_num=heads, merge=True, bias=False)


    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        context, _ = self.self_attn(inputs, inputs, inputs,
                                    mask=mask, attn_type="self")
        context = self.residual(context, inputs)
        output = self.feed_forward(context)
        return output

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

class LorentzCrossTransformerLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model=256, heads=8, d_ff=None, dropout=0.1, attention_dropout=0.1,
                 max_relative_positions=0):
        super(LorentzCrossTransformerLayer, self).__init__()

        if d_ff is None:
            d_ff = d_model * 4
        self.manifold = Lorentz()
        self.self_attn = LorentzMultiHeadedAttention(
            heads, d_model, self.manifold, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = LorentzPositionwiseFeedForward(d_model, d_ff, self.manifold, dropout=dropout)
        self.dropout = nn.Dropout(0.1)
        self.residual = LorentzLinear(d_model, d_model, dropout=dropout, head_num=heads, merge=True, bias=False)


    def forward(self, K, V, Q, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        context, _ = self.self_attn(K, V, Q,
                                    mask=mask, attn_type="self")
        context = self.residual(context, Q)
        output = self.feed_forward(context)
        return output

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout

class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ClosedLoopModel(nn.Module):
    def __init__(self, modelConfig, beta_1, beta_T, T, t_in, a_in, v_in, d_m, dropout, label_dim,
                 unified_size, vertex_num, routing, T_t, T_a, T_v,  batch_size):
        super().__init__()

        self.T = T
        self.batch_size = batch_size
        self.mult_dropout = dropout

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # Feature Extraction
        self.fc_pre_t_1 = nn.LSTM(t_in, modelConfig["t_in_pre"], bidirectional=True)
        # self.manifold1 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_pre_t_2 = LorentzFullyConnected(in_features=modelConfig["t_in_pre"]*2, out_features=modelConfig["t_in_pre"], manifold=self.manifold1, normalize=True)
        self.fc_pre_t_2 = nn.Linear(modelConfig["t_in_pre"] * 2, modelConfig["t_in_pre"])
        # self.manifold2 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_pre_v = LorentzFullyConnected(in_features=v_in, out_features=modelConfig["v_in_pre"], manifold=self.manifold2, normalize=True)
        self.fc_pre_v = torch.nn.Linear(v_in, modelConfig["v_in_pre"])
        # self.manifold3 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_pre_com = nn.Sequential(LorentzFullyConnected(in_features=modelConfig["t_in"], out_features=unified_size, manifold=self.manifold3, normalize=True), torch.nn.ReLU(), nn.Dropout(p=modelConfig["comments_dropout"]))
        self.fc_pre_com = nn.Sequential(torch.nn.Linear(modelConfig["t_in"], unified_size), torch.nn.ReLU(), nn.Dropout(p=modelConfig["comments_dropout"]))
        # self.manifold4 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_pre_user = nn.Sequential(LorentzFullyConnected(in_features=modelConfig["t_in"], out_features=unified_size, manifold=self.manifold4, normalize=True), torch.nn.ReLU(), nn.Dropout(p=modelConfig["comments_dropout"]))
        self.fc_pre_user = nn.Sequential(torch.nn.Linear(modelConfig["t_in"], unified_size), torch.nn.ReLU(), nn.Dropout(p=modelConfig["comments_dropout"]))
        #self.manifold5 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_pre_c3d = LorentzFullyConnected(in_features=modelConfig["c3d_in"], out_features=unified_size, manifold=self.manifold5, normalize=True)
        self.fc_pre_c3d = torch.nn.Linear(modelConfig["c3d_in"], unified_size)
        self.fc_pre_gpt_1 = nn.LSTM(t_in, modelConfig["t_in_pre"], bidirectional=True)
        #self.manifold6 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_pre_gpt_2 = LorentzFullyConnected(in_features=modelConfig["t_in_pre"] * 2, out_features=modelConfig["t_in_pre"], manifold=self.manifold6, normalize=True)
        self.fc_pre_gpt_2 = nn.Linear(modelConfig["t_in_pre"] * 2, modelConfig["t_in_pre"])

        #self.manifold7 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_pre_gpt_3 = LorentzFullyConnected(in_features=modelConfig["t_in_pre"], out_features=modelConfig["t_in_pre"], manifold=self.manifold7, normalize=True)
        self.fc_pre_gpt_3 = nn.Linear(modelConfig["t_in_pre"], modelConfig["t_in_pre"])
        self.vggish_layer = torch.hub.load('EmotionDiffusion-FakeSV/torchvggish-master', 'vggish', source='local')
        net_structure = list(self.vggish_layer.children())
        modified_fc_layers = list(net_structure[1].children())
        self.vggish_modified = nn.Sequential(modified_fc_layers[2], modified_fc_layers[3])
        # self.vggish_modified = nn.Sequential(*net_structure[-2:-1])
        #self.manifold8 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_pre_a = LorentzFullyConnected(in_features=a_in, out_features=modelConfig["a_in_pre"], manifold=self.manifold8, normalize=True)
        self.fc_pre_a = nn.Linear(a_in, modelConfig["a_in_pre"])

        # Intra-modal Enhancement
        #self.manifold9 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_t_MTout = LorentzFullyConnected(in_features=d_m * 3, out_features=128, manifold=self.manifold9, normalize=False)
        self.fc_t_MTout = nn.Linear(d_m * 3, modelConfig["t_in_pre"])
        #self.manifold10 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_g_MTout = LorentzFullyConnected(in_features=d_m * 3, out_features=128, manifold=self.manifold10, normalize=False)
        self.fc_g_MTout = nn.Linear(d_m * 3, modelConfig["t_in_pre"])
        #self.manifold11 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_a_MTout = LorentzFullyConnected(in_features=d_m * 3, out_features=d_m, manifold=self.manifold11, normalize=False)
        self.fc_a_MTout = nn.Linear(d_m * 3, d_m)
        #self.manifold12 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_v_MTout = LorentzFullyConnected(in_features=d_m * 3, out_features=d_m, manifold=self.manifold11, normalize=False)
        self.fc_v_MTout = nn.Linear(d_m * 3, d_m)

        #self.manifold13 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_g_t = LorentzFullyConnected(in_features=d_m * 3*2, out_features=d_m, manifold=self.manifold13, normalize=True)
        self.fc_g_t = nn.Linear(d_m * 3 * 2, d_m)
        #self.manifold14 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_a_MTout_dim = LorentzFullyConnected(in_features=d_m * 3, out_features=d_m, manifold=self.manifold14, normalize=True)
        self.fc_a_MTout_dim = nn.Linear(d_m * 3, d_m)
        #self.manifold15 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_v_MTout_dim = LorentzFullyConnected(in_features=d_m * 3, out_features=d_m, manifold=self.manifold15, normalize=True)
        self.fc_v_MTout_dim = nn.Linear(d_m * 3, d_m)

        self.CrossmodalTransformer = MULTModel(modelConfig["t_in_pre"], modelConfig["a_in_pre"], modelConfig["v_in_pre"], d_m, self.mult_dropout)

        self.LorentzCrossmodalTransformer_t_av = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)
        self.LorentzCrossmodalTransformer_a_tv = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)
        self.LorentzCrossmodalTransformer_v_ta = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)

        self.LorentzCrossmodalTransformer_g_tt = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)
        self.LorentzCrossmodalTransformer_g_aa = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)
        self.LorentzCrossmodalTransformer_g_vv = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)

        self.LorentzCrossmodalTransformer_a_tt = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)
        self.LorentzCrossmodalTransformer_a_gg = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)
        self.LorentzCrossmodalTransformer_a_vv = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)
        self.LorentzCrossmodalTransformer_v_tt = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)
        self.LorentzCrossmodalTransformer_v_gg = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)
        self.LorentzCrossmodalTransformer_v_aa = LorentzCrossTransformerLayer(d_model=128, heads=2, d_ff=None, dropout=0.1, attention_dropout=0.1)

        self.StoG = CapsuleSequenceToGraph(d_m, unified_size, vertex_num, routing, T_t, T_a, T_v)
        #self.manifold16 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_t_dim = LorentzFullyConnected(in_features=modelConfig["t_in_pre"], out_features=d_m, manifold=self.manifold16, normalize=True)
        self.fc_t_dim = nn.Linear(modelConfig["t_in_pre"], d_m)

        #self.manifold17 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_g_dim = LorentzFullyConnected(in_features=modelConfig["t_in_pre"], out_features=d_m, manifold=self.manifold17, normalize=True)
        # self.manifold_G = CustomLorentz(k=0.1, learnable=True)
        # self.fc_g_dim = LorentzFullyConnected(in_features=modelConfig["t_in_pre"], out_features=d_m, manifold=self.manifold_G, normalize=True)
        self.fc_g_dim = nn.Linear(in_features=modelConfig["t_in_pre"], out_features=d_m)

        self.Tucker_t_g = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_t_a = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_t_c = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_g_t = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_g_a = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_g_c = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_a_t = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_a_g = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_a_c = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_c_t = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_c_g = TuckERLayer(d_m, d_m * 3, d_m)
        self.Tucker_c_a = TuckERLayer(d_m, d_m * 3, d_m)
        self.LSTM_t = nn.LSTM(d_m, d_m, bidirectional=True)
        self.LSTM_a = nn.LSTM(d_m, d_m, bidirectional=True)
        self.LSTM_v = nn.LSTM(d_m, d_m, bidirectional=True)
        self.f_t_dim = nn.Linear(256, d_m)
        self.f_a_dim = nn.Linear(256, d_m)
        self.f_v_dim = nn.Linear(256, d_m)
        self.self_attn_t = TransformerBlock(d_model=d_m * 2, nhead=8, dropout=0.2)
        self.self_attn_a = TransformerBlock(d_model=d_m * 2, nhead=8, dropout=0.2)
        self.self_attn_v = TransformerBlock(d_model=d_m * 2, nhead=8, dropout=0.2)


        self.manifold1 = CustomLorentz(k=0.1, learnable=True)

        self.Lorentz_map_t = LorentzFullyConnected(in_features=d_m, out_features=d_m, manifold=self.manifold1, normalize=True)
        self.Lorentz_map_a = LorentzFullyConnected(in_features=d_m, out_features=d_m, manifold=self.manifold1, normalize=True)
        self.Lorentz_map_v = LorentzFullyConnected(in_features=d_m, out_features=d_m, manifold=self.manifold1, normalize=True)

        self.conv_t = LorentzConv1d(manifold=self.manifold1, in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, LFC_normalize=True)
        self.conv_a = LorentzConv1d(manifold=self.manifold1, in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, LFC_normalize=True)
        self.conv_v = LorentzConv1d(manifold=self.manifold1, in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, LFC_normalize=True)

        self.Lorentz_attn_t = LorentzTransformerEncoderLayer(d_model=d_m, heads=8, d_ff=None, dropout=0.2, attention_dropout=0.1)
        self.Lorentz_attn_a = LorentzTransformerEncoderLayer(d_model=d_m, heads=8, d_ff=None, dropout=0.2, attention_dropout=0.1)
        self.Lorentz_attn_v = LorentzTransformerEncoderLayer(d_model=d_m, heads=8, d_ff=None, dropout=0.2, attention_dropout=0.1)
        # Cross-modal Interaction
        self.model_t = Text_Noise_Pre(T=modelConfig["T"], ch=modelConfig["vertex_num"], dropout=modelConfig["Text_Pre_dropout"], in_ch=unified_size)
        self.model_a = Audio_Noise_Pre(T=modelConfig["T"], ch=modelConfig["vertex_num"], dropout=modelConfig["Img_Pre_dropout"], in_ch=unified_size)
        self.model_v = Visual_Noise_Pre(T=modelConfig["T"], ch=modelConfig["vertex_num"], dropout=modelConfig["Img_Pre_dropout"], in_ch=unified_size)

        # self.manifold4 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_t = LorentzFullyConnected(in_features=vertex_num, out_features=1, manifold=self.manifold4, normalize=False)
        # self.manifold5 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_a = LorentzFullyConnected(in_features=vertex_num, out_features=1, manifold=self.manifold5, normalize=False)
        # self.manifold6 = CustomLorentz(k=1.0, learnable=True)
        # self.fc_v = LorentzFullyConnected(in_features=vertex_num, out_features=1, manifold=self.manifold6, normalize=False)

        self.fc_t = nn.Linear(in_features=vertex_num, out_features=1)
        self.fc_a = nn.Linear(in_features=vertex_num, out_features=1)
        self.fc_v = nn.Linear(in_features=vertex_num, out_features=1)

        self.fc_t_ = nn.Sequential(torch.nn.Linear(d_m-1, d_m), nn.Dropout(p=0.3))
        self.fc_a_ = nn.Sequential(torch.nn.Linear(d_m-1, d_m), nn.Dropout(p=0.3))
        self.fc_v_ = nn.Sequential(torch.nn.Linear(d_m-1, d_m), nn.Dropout(p=0.3))

        self.linear_a = torch.nn.Linear(128,128*4)
        self.linear_t = torch.nn.Linear(128,128*4)
        self.linear_v = torch.nn.Linear(128,128*4)
        self.sum_pool = torch.nn.Conv1d(1,1,4,4)
        self.scale_kappa = nn.Parameter(torch.tensor(0.001))


        self.fc_m = nn.Linear(in_features=unified_size * 3, out_features=unified_size)


        # Prediction
        self.fc_pre = nn.Linear(in_features=unified_size, out_features=label_dim)
        self.trm = nn.TransformerEncoderLayer(d_model=unified_size, nhead=2, batch_first=True)

        self.Lorentz_map_last = LorentzFullyConnected(in_features=d_m, out_features=d_m, manifold=self.manifold1, normalize=True)

        # self.predictor = LorentzMLR(num_features=unified_size, num_classes=label_dim, manifold=self.manifold)
        self.predictor = LorentzMLR(num_features=128, num_classes=2, manifold=self.manifold1)

    def contrastive_loss_two_modal(self, v_embed, t_embed):
        v_embed = torch.mean(v_embed, dim=1, keepdim=False)
        t_embed = torch.mean(t_embed, dim=1, keepdim=False)
        v_embed = F.normalize(v_embed, dim=1)
        t_embed = F.normalize(t_embed, dim=1)
        pos_vt = torch.sum(v_embed * t_embed, dim=1, keepdim=True)
        neg_v = torch.matmul(v_embed, v_embed.t())
        neg_t = torch.matmul(t_embed, t_embed.t())
        neg_v = neg_v - torch.diag_embed(torch.diag(neg_v))
        neg_t = neg_t - torch.diag_embed(torch.diag(neg_t))
        pos = torch.mean(pos_vt, dim=1)
        bsz = neg_v.size(0)
        neg = torch.sum(torch.cat([neg_v, neg_t], dim=1), dim=1)
        neg = neg / (2*(bsz - 1))
        loss = torch.mean(F.softplus(neg - pos))
        return loss

    def contrastive_loss_three_modal(self, text_embed, audio_embed, video_embed):
        # -------- Step 1: 在模态内做平均 (如维度是 [B, T, D] -> [B, D]) --------
        text_embed = torch.mean(text_embed, dim=1, keepdim=False)
        audio_embed = torch.mean(audio_embed, dim=1, keepdim=False)
        video_embed = torch.mean(video_embed, dim=1, keepdim=False)

        # -------- Step 2: L2归一化 --------
        text_embed = text_embed / torch.norm(text_embed, dim=1, keepdim=True)
        audio_embed = audio_embed / torch.norm(audio_embed, dim=1, keepdim=True)
        video_embed = video_embed / torch.norm(video_embed, dim=1, keepdim=True)

        # -------- Step 3: 正样本相似度（同索引对之间） --------
        pos_text_audio = torch.sum(text_embed * audio_embed, dim=1, keepdim=True)
        pos_text_video = torch.sum(text_embed * video_embed, dim=1, keepdim=True)
        pos_audio_video = torch.sum(audio_embed * video_embed, dim=1, keepdim=True)

        # -------- Step 4: 平均正样本相似度 --------
        pos = torch.mean(torch.cat([pos_text_audio, pos_text_video, pos_audio_video], dim=1), dim=1)  # shape: (B,)

        # -------- Step 5: 构造负样本相似度矩阵（去掉对角线） --------
        neg_text = torch.matmul(text_embed, text_embed.T)
        neg_audio = torch.matmul(audio_embed, audio_embed.T)
        neg_video = torch.matmul(video_embed, video_embed.T)

        neg_text = neg_text - torch.diag_embed(torch.diag(neg_text))
        neg_audio = neg_audio - torch.diag_embed(torch.diag(neg_audio))
        neg_video = neg_video - torch.diag_embed(torch.diag(neg_video))

        # -------- Step 6: 平均负样本相似度 --------
        neg = torch.mean(torch.cat([neg_text, neg_audio, neg_video], dim=1), dim=1)  # (B,)

        # -------- Step 7: 计算对比损失 --------
        loss = torch.mean(F.softplus(neg - pos))  # scalar

        return loss

    def forward(self, texts, audios, videos, comments, c3d, user_intro, gpt_description):
        # Feature Extraction
        texts_local, _ = self.fc_pre_t_1(texts)
        texts_local = self.fc_pre_t_2(texts_local)
        audios = self.vggish_modified(audios)
        audios_local = self.fc_pre_a(audios)
        c3d_local = self.fc_pre_c3d(c3d)
        gpt_local, _ = self.fc_pre_gpt_1(gpt_description)
        gpt_local = self.fc_pre_gpt_2(gpt_local)
        gpt_local = self.fc_pre_gpt_3(gpt_local)
        comments_global = self.fc_pre_com(comments)
        user_intro_global = self.fc_pre_user(user_intro.squeeze())
        videos = self.fc_pre_v(videos)
        videos_global = torch.mean(videos, -2)

        # Intra-modal Enhancement  ------texts_local (16,1,100), gpt_local (16,1,100), audios_local (16,50,128), c3d_local (16,83,128)-----


        texts_local = self.fc_t_dim(texts_local)
        gpt_local = self.fc_g_dim(gpt_local)
        z_t_g = self.Tucker_t_g(texts_local, gpt_local)
        z_t_a = self.Tucker_t_a(texts_local, audios_local)
        z_t_c = self.Tucker_t_c(texts_local, c3d_local)
        z_g_t = self.Tucker_g_t(gpt_local, texts_local)
        z_g_a = self.Tucker_g_a(gpt_local, audios_local)
        z_g_c = self.Tucker_g_c(gpt_local, c3d_local)
        z_a_t = self.Tucker_a_t(audios_local, texts_local)
        z_a_g = self.Tucker_a_g(audios_local, gpt_local)
        z_a_c = self.Tucker_a_c(audios_local, c3d_local)
        z_c_t = self.Tucker_c_t(c3d_local, texts_local)
        z_c_g = self.Tucker_c_g(c3d_local, gpt_local)
        z_c_a = self.Tucker_c_a(c3d_local, audios_local)
        # z_t (1,16,384), z_g (1,16,384 ), z_a (50,16,384), z_v (83, 16, 384)
        z_t = torch.cat([z_t_g, z_t_a, z_t_c], dim=2)
        z_g = torch.cat([z_g_t, z_g_a, z_g_c], dim=2)
        z_a = torch.cat([z_a_t, z_a_g, z_a_c], dim=2)
        z_v = torch.cat([z_c_t, z_c_g, z_c_a], dim=2)

        z_t = self.fc_t_MTout(z_t)
        z_g = self.fc_g_MTout(z_g)
        z_a = self.fc_a_MTout(z_a)
        z_v = self.fc_v_MTout(z_v)
        # -------------------↓↓↓----Dimension transformation----↓↓↓---------------------
        z_t = z_t.permute(1, 0, 2)
        z_g = z_g.permute(1, 0, 2)
        z_a = z_a.permute(1, 0, 2)
        z_v = z_v.permute(1, 0, 2)
        # -------------------↑↑↑------------------------------- ↑↑↑----------------------
        # z_t_gg = self.LorentzCrossmodalTransformer_t_gg(z_g, z_g, z_t, mask=None)       # Hrer input are K, V, Q;  Not conventional Q, K, V
        # z_t_aa = self.LorentzCrossmodalTransformer_t_aa(z_a, z_a, z_t, mask=None)
        # z_t_vv = self.LorentzCrossmodalTransformer_t_vv(z_v, z_v, z_t, mask=None)
        # z_t_avg = F.dropout(torch.cat([z_t_gg, z_t_aa, z_t_vv], dim=2), p=self.mult_dropout, training=self.training)
        # z_t_avg = z_t_avg.transpose(0, 1)
        #
        # z_g_tt = self.LorentzCrossmodalTransformer_g_tt(z_t, z_t, z_g, mask=None)
        # z_g_aa = self.LorentzCrossmodalTransformer_g_aa(z_a, z_a, z_g, mask=None)
        # z_g_vv = self.LorentzCrossmodalTransformer_g_vv(z_v, z_v, z_g, mask=None)
        # z_g_tav = F.dropout(torch.cat([z_g_tt, z_g_aa, z_g_vv], dim=2), p=self.mult_dropout, training=self.training)
        # z_g_tav = z_g_tav.transpose(0, 1)
        #
        # z_a_tt = self.LorentzCrossmodalTransformer_a_tt(z_t, z_t, z_a, mask=None)
        # z_a_gg = self.LorentzCrossmodalTransformer_a_gg(z_g, z_g, z_a, mask=None)
        # z_a_vv = self.LorentzCrossmodalTransformer_a_vv(z_v, z_v, z_a, mask=None)
        # z_a_tvg = F.dropout(torch.cat([z_a_tt, z_a_gg, z_a_vv], dim=2), p=self.mult_dropout, training=self.training)
        # z_a_tvg = z_a_tvg.transpose(0, 1)
        #
        # z_v_tt = self.LorentzCrossmodalTransformer_v_tt(z_t, z_t, z_v, mask=None)
        # z_v_gg = self.LorentzCrossmodalTransformer_v_gg(z_g, z_g, z_v, mask=None)
        # z_v_aa = self.LorentzCrossmodalTransformer_v_aa(z_a, z_a, z_v, mask=None)
        # z_v_tag = F.dropout(torch.cat([z_v_tt, z_v_gg, z_v_aa], dim=2), p=self.mult_dropout, training=self.training)
        # z_v_tag = z_v_tag.transpose(0, 1)


        z_t_avg, z_g_tav, z_a_tvg, z_v_tag = self.CrossmodalTransformer(z_t, z_g, z_a, z_v)
        # -------------------↓↓↓----contrastive loss----↓↓↓---------------------
        z_t_avg_cl = z_t_avg.transpose(0, 1)
        z_g_tav_cl = z_g_tav.transpose(0, 1)
        loss_cl_two = self.contrastive_loss_two_modal(z_t_avg_cl, z_g_tav_cl)
        z_a_tvg_cl = z_a_tvg.transpose(0, 1)
        z_v_tag_cl = z_v_tag.transpose(0, 1)
        loss_cl_three = self.contrastive_loss_three_modal(z_t_avg_cl, z_a_tvg_cl, z_v_tag_cl)
        loss_cl = loss_cl_two + loss_cl_three
        # -------------------↑↑↑------------------------------- ↑↑↑----------------------

        z_t = self.fc_g_t(torch.cat([z_t_avg, z_g_tav], dim=2))
        z_a = self.fc_a_MTout_dim(z_a_tvg)
        z_v = self.fc_v_MTout_dim(z_v_tag)

        feature_a = torch.mean(z_a.transpose(0,1), dim=1, keepdim=True)
        feature_t = torch.mean(z_t.transpose(0,1), dim=1, keepdim=True)
        feature_v = torch.mean(z_v.transpose(0,1), dim=1, keepdim=True)

        fea_a = self.linear_a(feature_a)
        fea_t = self.linear_t(feature_t)
        fea_v = self.linear_v(feature_v)

        fea_a = F.tanh(fea_a)
        fea_t = F.tanh(fea_t)
        fea_v = F.tanh(fea_v)
        fea_atv_ = torch.mul(torch.mul(fea_t,fea_a),fea_v)
        fea_atv = (self.sum_pool(fea_atv_) + feature_a + feature_t + feature_v)
        fea_atv = self.fc_pre(torch.mean(fea_atv, -2))

        x_t, x_a, x_v = self.StoG(z_t, z_a, z_v, self.batch_size) # x_t (16, 32, 128), x_a (16, 32, 128), x_v (16, 32, 128)
        #----------cNN-----↓↓↓------
        # -----------------↑↑↑------
        # x_t_pre, _ = self.LSTM_t(x_t)
        # x_a_pre, _ = self.LSTM_a(x_a)
        # x_v_pre, _ = self.LSTM_v(x_v)

        x_t = self.manifold1.expmap0(x_t)
        x_a = self.manifold1.expmap0(x_a)
        x_v = self.manifold1.expmap0(x_v)


        x_t = self.Lorentz_map_t(x_t)
        x_a = self.Lorentz_map_a(x_a)
        x_v = self.Lorentz_map_v(x_v)

        # x_t_aa = self.LorentzCrossmodalTransformer_t_gg(x_a, x_a, x_t, mask=None)       # Hrer input are K, V, Q;  Not conventional Q, K, V
        # x_t_vv = self.LorentzCrossmodalTransformer_t_aa(x_v, x_v, x_t, mask=None)
        # z_t_av = F.dropout(torch.cat([x_t_aa, x_t_vv], dim=2), p=self.mult_dropout, training=self.training)
        #
        # z_a_tt = self.LorentzCrossmodalTransformer_g_tt(x_t, x_t, x_a, mask=None)
        # z_a_vv = self.LorentzCrossmodalTransformer_g_aa(x_v, x_v, x_a, mask=None)
        # z_a_tv = F.dropout(torch.cat([z_a_tt, z_a_vv], dim=2), p=self.mult_dropout, training=self.training)
        #
        # z_v_tt = self.LorentzCrossmodalTransformer_a_tt(x_t, x_t, x_v, mask=None)
        # z_v_aa = self.LorentzCrossmodalTransformer_a_gg(x_a, x_a, x_v, mask=None)
        # z_v_ta = F.dropout(torch.cat([z_v_tt, z_v_aa], dim=2), p=self.mult_dropout, training=self.training)

        z_t_av = self.LorentzCrossmodalTransformer_t_av(x_a, x_v, x_t, mask=None)       # Hrer input are K, V, Q;  Not conventional Q, K, V

        z_a_tv = self.LorentzCrossmodalTransformer_a_tv(x_t, x_v, x_a, mask=None)

        z_v_ta = self.LorentzCrossmodalTransformer_v_ta(x_t, x_a, x_v, mask=None)

        # x_t_pre = self.conv_t(z_t_av)
        # x_a_pre = self.conv_a(z_a_tv)
        # x_v_pre = self.conv_v(z_v_ta)

        x_t_pre = self.Lorentz_attn_t(z_t_av, None)
        x_a_pre = self.Lorentz_attn_a(z_a_tv, None)
        x_v_pre = self.Lorentz_attn_v(z_v_ta, None)

        x_t_pre = self.manifold1.logmap0(x_t_pre)[..., 1:]
        x_a_pre = self.manifold1.logmap0(x_a_pre)[..., 1:]
        x_v_pre = self.manifold1.logmap0(x_v_pre)[..., 1:]

        output_a = self.fc_a(x_a_pre.transpose(2,1))
        output_t = self.fc_t(x_t_pre.transpose(2,1))
        output_v = self.fc_v(x_v_pre.transpose(2,1))

        output_a = self.fc_a_(output_a.transpose(2, 1)) #(17, 1, 128)
        output_t = self.fc_t_(output_t.transpose(2, 1))
        output_v = self.fc_v_(output_v.transpose(2, 1))

        # Prediction
        output_m = torch.concat([output_t, output_a, output_v, gpt_local], dim=1)
        output_m = self.trm(output_m)
        output_m = torch.mean(output_m, -2)

        output_m = self.manifold1.expmap0(output_m)
        output_m = self.Lorentz_map_last(output_m)
        output_m = self.predictor(output_m.squeeze())

        return loss_cl, output_m, fea_atv