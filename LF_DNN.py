import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubNet(nn.Module):

    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=True):
        super(SubNet, self).__init__()
        if num_layers == 1:
            dropout = 0.0
        self.rnn = nn.GRU(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        _, final_states = self.rnn(x)
        h = torch.cat((final_states[0], final_states[1]), -1)
        dropped = self.dropout(h)
        y_1 = F.relu(self.linear_1(dropped), inplace=True)
        y_2 = F.relu(self.linear_2(y_1), inplace=True)
        return y_2


class ExpressionSelfAttention(nn.Module):
    """Paper equations (4)-(7): local visual and global-relation attention."""

    def __init__(self, in_size, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(in_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(2 * in_size, 1)
        self.linear_3 = nn.Linear(2 * in_size, in_size)

    def forward(self, x):
        # x: [phrases, visual_dim]
        alpha = torch.sigmoid(self.linear_1(self.dropout(x)))
        global_visual = (alpha * x).sum(dim=0, keepdim=True) / alpha.sum(dim=0, keepdim=True).clamp_min(1e-8)
        global_visual = global_visual.expand(x.size(0), -1)
        visual_context = torch.cat([x, global_visual], dim=-1)
        beta = torch.sigmoid(self.linear_2(self.dropout(visual_context)))
        # Eq. (7): concatenate the weighted local visual feature and global feature.
        filtered = torch.cat([alpha * beta * x, global_visual], dim=-1)
        return F.relu(self.linear_3(filtered))


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(CrossModalAttention, self).__init__()
        self.activation = nn.ReLU()
        self.x_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.y_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.t_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.x_weight_1.data.fill_(1)
        self.y_weight_1.data.fill_(1)
        self.t_weight_1.data.fill_(1)
        self.bias.data.fill_(0)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, y, target):
        x_att = torch.matmul(x, x.transpose(-1, -2))
        x_att = self.activation(x_att)

        y_att = torch.matmul(y, y.transpose(-1, -2))
        y_att = self.activation(y_att)

        t_att = torch.matmul(target, target.transpose(-1, -2))
        t_att = self.activation(t_att)

        fusion_att = self.x_weight_1 * x_att + self.y_weight_1 * y_att + self.t_weight_1 * t_att + self.bias
        fusion_att = nn.Softmax(dim=-1)(fusion_att)
        target_att = torch.matmul(fusion_att, target)

        dropped = self.dropout(target_att)
        y_1 = F.relu(self.linear_1(dropped), inplace=True)
        y_2 = F.relu(self.linear_2(y_1), inplace=True)

        return y_2


class FusionSubNet(nn.Module):
    """Paper equations (13)-(14): MIL with Bi-GRU, tanh MLP and Softmax."""

    def __init__(self, in_size, out_size, dropout):
        super().__init__()
        self.rnn = nn.GRU(in_size, in_size, num_layers=1, dropout=0,
                          bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(2 * in_size, in_size)
        self.linear_2 = nn.Linear(in_size, 1, bias=False)

    def forward(self, h, p):
        # h: [phrases, fusion_dim], p: [phrases, output_dim].
        output, _ = self.rnn(h.unsqueeze(0))
        attention_hidden = torch.tanh(self.linear_1(self.dropout(output)))
        logits = self.linear_2(attention_hidden).squeeze(-1)  # [1, phrases]
        gamma = torch.softmax(logits, dim=1)
        return (gamma.unsqueeze(-1) * p.unsqueeze(0)).sum(dim=1).squeeze(0)


class LF_DNN(nn.Module):

    def __init__(self, args):
        super(LF_DNN, self).__init__()
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims
        self.text_out = args.text_out

        self.audio_prob, self.video_prob, self.text_prob = args.dropouts
        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts

        self.post_fusion_dim = args.post_fusion_dim
        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim

        self.output_dim = args.num_classes if args.train_mode == "classification" else 1

        # original: paper-aligned FGMSA; darf: identical backbone plus residual DARF.
        self.relation_mode = os.environ.get("FGMSA_RELATION_MODE", "original").lower()
        if self.relation_mode not in ("original", "darf"):
            raise ValueError("FGMSA_RELATION_MODE must be 'original' or 'darf'")

        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, dropout=self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, dropout=self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, dropout=self.text_prob)

        self.video_attnet = ExpressionSelfAttention(self.video_in, self.video_prob)

        self.audio_linear = nn.Linear(self.audio_in, self.audio_hidden)
        self.video_linear = nn.Linear(self.video_in, self.video_hidden)
        self.text_linear = nn.Linear(self.text_in, self.text_hidden)
        self.audio_cutnet = CrossModalAttention(self.audio_hidden, self.audio_prob)
        self.video_cutnet = CrossModalAttention(self.video_hidden, self.video_prob)
        self.text_cutnet = CrossModalAttention(self.text_hidden, self.text_prob)

        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_hidden + self.video_hidden + self.audio_hidden,
                                             self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_dim)

        self.fusion_subnet = FusionSubNet(self.post_fusion_dim, self.output_dim, self.post_fusion_prob)

        if self.relation_mode == "darf":
            # Project MAM outputs into one comparison space. The output of
            # each CrossModalAttention is [phrases, modality_hidden_dim].
            compare_dim = min(self.text_hidden, self.audio_hidden, self.video_hidden)
            self.darf_audio_proj = nn.Linear(self.audio_hidden, compare_dim)
            self.darf_video_proj = nn.Linear(self.video_hidden, compare_dim)
            self.darf_text_proj = nn.Linear(self.text_hidden, compare_dim)
            self.darf_correction = nn.Sequential(
                nn.Linear(self.post_fusion_dim + 3, self.post_fusion_dim),
                nn.ReLU(),
                nn.Linear(self.post_fusion_dim, self.post_fusion_dim),
                nn.Tanh(),
            )
            # Small initialization prevents overwriting the pre-existing fusion.
            self.darf_gate_logit = nn.Parameter(torch.tensor(-4.0))

        self.post_text_dropout = nn.Dropout(p=self.post_text_prob)
        self.post_text_layer_1 = nn.Linear(self.text_hidden, self.post_text_dim)
        self.post_text_layer_2 = nn.Linear(self.post_text_dim, self.post_text_dim)
        self.post_text_layer_3 = nn.Linear(self.post_text_dim, self.output_dim)

        self.post_audio_dropout = nn.Dropout(p=self.post_audio_prob)
        self.post_audio_layer_1 = nn.Linear(self.audio_hidden, self.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(self.post_audio_dim, self.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(self.post_audio_dim, self.output_dim)

        self.post_video_dropout = nn.Dropout(p=self.post_video_prob)
        self.post_video_layer_1 = nn.Linear(self.video_hidden, self.post_video_dim)
        self.post_video_layer_2 = nn.Linear(self.post_video_dim, self.post_video_dim)
        self.post_video_layer_3 = nn.Linear(self.post_video_dim, self.output_dim)

    def forward(self, text_cutx, audio_cutx, video_cutx):
        audio_batchs = list()
        video_batchs = list()
        text_batchs = list()
        fusion_batchs = list()
        for i in range(len(text_cutx)):
            audio_cutxi = audio_cutx[i]
            video_cutxi = video_cutx[i]
            text_cutxi = text_cutx[i]

            audio_hi = self.audio_subnet(audio_cutxi)
            video_hi = self.video_subnet(video_cutxi)
            text_hi = self.text_subnet(text_cutxi)
            audio_batchs.append(audio_hi.unsqueeze(0))
            video_batchs.append(video_hi.unsqueeze(0))
            text_batchs.append(text_hi.unsqueeze(0))

            video_cutxi = self.video_attnet(video_cutxi)

            audio_cutxi = self.audio_linear(audio_cutxi)
            video_cutxi = self.video_linear(video_cutxi)
            text_cutxi = self.text_linear(text_cutxi)
            audio_cuth = self.audio_cutnet(video_cutxi, text_cutxi, audio_cutxi)
            video_cuth = self.video_cutnet(audio_cutxi, text_cutxi, video_cutxi)
            text_cuth = self.text_cutnet(audio_cutxi, video_cutxi, text_cutxi)

            fusion_h = torch.cat([audio_cuth, video_cuth, text_cuth], dim=-1)
            dropped = self.post_fusion_dropout(fusion_h)
            x_1 = F.relu(self.post_fusion_layer_1(dropped), inplace=True)
            if self.relation_mode == "darf":
                za = F.normalize(self.darf_audio_proj(audio_cuth), p=2, dim=-1, eps=1e-8)
                zv = F.normalize(self.darf_video_proj(video_cuth), p=2, dim=-1, eps=1e-8)
                zt = F.normalize(self.darf_text_proj(text_cuth), p=2, dim=-1, eps=1e-8)
                # Phrase-wise pairwise cosine distances: [L, 3].
                disagreements = torch.stack([
                    1.0 - (zt * za).sum(dim=-1),
                    1.0 - (zt * zv).sum(dim=-1),
                    1.0 - (za * zv).sum(dim=-1),
                ], dim=-1)
                correction = self.darf_correction(torch.cat([x_1, disagreements], dim=-1))
                x_1 = x_1 + torch.sigmoid(self.darf_gate_logit) * correction
            x_2 = F.relu(self.post_fusion_layer_2(x_1), inplace=True)
            _fusion = self.post_fusion_layer_3(x_2)
            fusion = self.fusion_subnet(x_1, _fusion)
            fusion_batchs.append(fusion.unsqueeze(0))
        audio_h = torch.cat(audio_batchs, dim=0)
        video_h = torch.cat(video_batchs, dim=0)
        text_h = torch.cat(text_batchs, dim=0)
        fusions = torch.cat(fusion_batchs, dim=0)
        output_fusion = fusions.view(-1, self.output_dim)

        x_t = self.post_text_dropout(text_h)
        x_t = F.relu(self.post_text_layer_1(x_t), inplace=True)
        x_t = F.relu(self.post_text_layer_2(x_t), inplace=True)
        output_text = self.post_text_layer_3(x_t)

        x_a = self.post_audio_dropout(audio_h)
        x_a = F.relu(self.post_audio_layer_1(x_a), inplace=True)
        x_a = F.relu(self.post_audio_layer_2(x_a), inplace=True)
        output_audio = self.post_audio_layer_3(x_a)

        x_v = self.post_video_dropout(video_h)
        x_v = F.relu(self.post_video_layer_1(x_v), inplace=True)
        x_v = F.relu(self.post_video_layer_2(x_v), inplace=True)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,

            'M': output_fusion,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return res
