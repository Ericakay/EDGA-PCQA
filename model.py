import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from util import sample_and_group


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim,
                                         num_heads=num_heads,
                                         batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):

        seq = torch.stack([x, y], dim=1)  # [B, 2, D]
        attn_out, _ = self.mha(seq, seq, seq)  # [B, 2, D]
        attn_out = self.norm(attn_out + seq)
        fused = attn_out.mean(dim=1)  # [B, D]
        return fused


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # ([batchsize, npoints, neighbor, feature])
        x = x.permute(0, 1, 3, 2)  # ([batchsize, npoints, feature, neighbor])
        x = x.reshape(-1, d, s)  # ([batchsize*npoints, feature, neighbor])
        batch_size, _, N = x.size()  # ([batchsize*npoints, feature, neighbor])
        x1 = F.relu(self.bn1(self.conv1(x)))  # ([batchsize*npoints, feature, neighbor])
        x2 = F.relu(self.bn2(self.conv2(x1)))  # ([batchsize*npoints, feature, neighbor])
        x3 = F.adaptive_max_pool1d(x2, 1)  # ([batchsize*npoints, feature, 1 ])
        x4 = x3.view(batch_size, -1)  # ([batchsize*npoints, feature])
        x_res = x4.reshape(b, n, -1).permute(0, 2, 1)
        return x_res  # ([batchsize, feature, npoints])


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, eval, drop_rate, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # device = torch.device('cuda')
    GorC = torch.cuda.is_available()
    device = torch.device("cuda" if GorC else "cpu")
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    if not eval and drop_rate > 0:
        mask = torch.rand(feature[:, :1, :, :].size(),
                          device=feature.device) > drop_rate
        feature[:, :6, :, :] *= mask.float()

    return feature


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=1):
        super().__init__()
        self.heads = heads
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features * heads, bias=False)
        self.attn = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        """x shape: [B, C, N, K] (batch, channels, points, neighbors)"""
        B, C, N, K = x.size()

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, N, K, C]
        x = x.view(B * N, K, C)  # [B*N, K, C]

        h = self.W(x)  # [B*N, K, out_features*heads]
        h = h.view(B * N, K, self.heads, self.out_features)
        h = h.permute(0, 2, 1, 3)  # [B*N, heads, K, out_features]

        center = h[:, :, 0:1, :]  # [B*N, heads, 1, out_features]

        energy = torch.cat([
            center.repeat(1, 1, K, 1),
            h
        ], dim=-1)  # [B*N, heads, K, 2*out_features]

        energy = self.attn(energy).squeeze(-1)  # [B*N, heads, K]
        energy = self.leaky_relu(energy)

        attention = F.softmax(energy, dim=-1)  # [B*N, heads, K]

        out = torch.matmul(
            attention.unsqueeze(-2),  # [B*N, heads, 1, K]
            h  # [B*N, heads, K, out_features]
        ).squeeze(-2)  # [B*N, heads, out_features]

        out = out.transpose(1, 2).contiguous()  # [B*N, out_features, heads]
        out = out.view(B * N, self.heads * self.out_features)

        out = out.view(B, N, -1).permute(0, 2, 1)  # [B, out_features*heads, N]
        return out

class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class EDGA(nn.Module):

    def __init__(self, args, output_channels=1):
        super(EDGA, self).__init__()
        self.args = args
        self.k = args.k

        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.gat1 = GATLayer(64, 64, heads=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.gat2 = GATLayer(64, 64, heads=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.gat3 = GATLayer(128, 128, heads=1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.gat4 = GATLayer(256, 256, heads=1)

        # 最终卷积层
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        batch_size = x.size(0)

     
        x = get_graph_feature(x, self.args.eval, drop_rate=self.args.q1, k=self.k)
        x = self.conv1(x)  # [B, 64, N, K]
        x = self.gat1(x)  # [B, 64, N]
        x1 = x

        
        x = get_graph_feature(x1, self.args.eval, drop_rate=self.args.q2, k=self.k)
        x = self.conv2(x)  # [B, 64, N, K]
        x = self.gat2(x)  # [B, 64, N]
        x2 = x

       
        x = get_graph_feature(x2, self.args.eval, drop_rate=self.args.q3, k=self.k)
        x = self.conv3(x)  # [B, 128, N, K]
        x = self.gat3(x)  # [B, 128, N]
        x3 = x

        
        x = get_graph_feature(x3, self.args.eval, drop_rate=self.args.q4, k=self.k)
        x = self.conv4(x)  # [B, 256, N, K]
        x = self.gat4(x)  # [B, 256, N]
        x4 = x

       
        x = torch.cat([x1, x2, x3, x4], dim=1)  # [B, 512, N]

      
        x = self.conv5(x)  # [B, emb_dims, N]
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        return torch.cat([x1, x2], 1)  # [B, emb_dims*2]


class Pct_3DTA(nn.Module):
    def __init__(self, args, final_channels=1):
        super(Pct_3DTA, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1, stride=int(args.point_num / 256), bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(1024)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.conv_fuse1 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.conv_fuse2 = nn.Sequential(nn.Conv1d(768, 1024, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(1024),
                                        nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        xyz = x[:, 0:3, :].permute(0, 2, 1)  # barch_size num_points xyz                # get xyz axis
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(
            self.conv1(x)))
        # B, D, N
        x_str = F.relu(self.bn2(self.conv2(x)))

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, neighbor=32, xyz=xyz,
                                                feature=x)  # SG new_feature[b,512,32,128]
        feature_0 = self.gather_local_0(new_feature)  # [B, 128, 512] <= [B, 512, 32, 128]     GL

        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, neighbor=32, xyz=new_xyz,
                                                feature=feature_0)  # SG [B, 256, 32, 256]
        feature_1 = self.gather_local_1(new_feature)  # [B, 256, 256] <= [B, 256, 32, 256]     GL

        feature_1 = torch.cat((feature_1, x_str), dim=1)
        feature_1 = self.conv_fuse1(feature_1)  # [b,256,256]
        x = F.adaptive_max_pool1d(feature_1, 1).view(batch_size, -1)  # CBR

        return x


class ChannelAttention(nn.Module):  # !!!
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c)
        y = self.fc(y)
        return x * y


class SpatialAttention(nn.Module):  # !!!
    def __init__(self, dim):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * attention


# class ToubleEDGA(nn.Module):
#         def __init__(self, args, output_channels=1):
#             super(ToubleEDGA, self).__init__()
#             self.args = args
#             self.EDGA = EDGA(args)
#             self.pct = Pct_3DTA(args)
#             self.linear1 = nn.Linear(2048+256, 512, bias=False) # args.emb_dims * 2+256
#             self.bn6 = nn.BatchNorm1d(512)
#             self.dp1 = nn.Dropout(p=args.dropout)
#             self.linear2 = nn.Linear(512, 256)
#             self.bn7 = nn.BatchNorm1d(256)
#             self.dp2 = nn.Dropout(p=args.dropout)
#             self.linear3 = nn.Linear(256, output_channels)
#             self.channel_att = ChannelAttention(256)
#             self.spatial_att = SpatialAttention(2048)
#
#         def forward(self, x):  # x: [32,6,1024]
#             y = x
#             x = self.pct(x) # final [32,256,256]  [32,256]
#             y = self.EDGA(y) # final [32,2048]
#             pct_att = self.channel_att(x)   
#             EDGA_att = self.spatial_att(y.unsqueeze(-1)).squeeze()  
#             x = torch.cat((pct_att, EDGA_att), dim=1)
#             x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
#             x = self.dp1(x)
#             x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
#             x = self.dp2(x)
#             x = self.linear3(x)
#
#             return x

class ToubleEDGA(nn.Module):
    def __init__(self, args, fusion_dim=512, output_channels=1):
        super().__init__()
        self.EDGA = EDGA(args)
        self.pct = Pct_3DTA(args)

        self.proj_pct = nn.Linear(256, fusion_dim, bias=False)
        self.proj_EDGA = nn.Linear(2048, fusion_dim, bias=False)

        self.fusion_attn = CrossModalAttention(dim=fusion_dim,
                                               num_heads=8)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: [B, 6, N]
        pct_feat = self.pct(x)  # [B, 256]
        EDGA_feat = self.EDGA(x)  # [B, 2048]

        x_p = self.proj_pct(pct_feat)  # [B, fusion_dim]
        y_p = self.proj_EDGA(EDGA_feat)  # [B, fusion_dim]

        fused = self.fusion_attn(x_p, y_p)  # [B, fusion_dim]

        out = self.classifier(fused)  # [B, output_channels]
        return out
