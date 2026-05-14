# import os, sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
from networks.blocks import MultiHeadTopoAttention, CrossAttention, AttnPool3d, DWSpatialGate3D, PartialDWRes3D
from utils.net_convert import *
# from torchinfo import summary


class DenseNet3DEncoder(nn.Module):
    """
    MONAI DenseNet121 3D backbone with a ResNet-like interface:
    exposes conv1/bn1/act/maxpool and layer1..layer4.

    It also projects the last stage feature from 1024 -> 2048 to keep
    the rest of the network unchanged (expected channels: 256,512,1024,2048).
    """
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.backbone = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=out_channels)

        f = self.backbone.features
        # map DenseNet stem to expected names used by _forward_stem
        self.conv1 = f.conv0
        self.bn1 = f.norm0
        self.act = f.relu0
        self.maxpool = f.pool0

        # stage-wise modules (DenseNet blocks + transitions)
        # stage1: denseblock1 -> 256
        # stage2: transition1 + denseblock2 -> 512
        # stage3: transition2 + denseblock3 -> 1024
        # stage4: transition3 + denseblock4 (+norm5+relu) -> 1024  (then project -> 2048)
        self.layer1 = nn.Sequential(
            f.denseblock1,  # -> 256
        )
        self.layer2 = nn.Sequential(
            f.transition1,  # downsample
            f.denseblock2,  # -> 512
        )
        self.layer3 = nn.Sequential(
            f.transition2,  # downsample
            f.denseblock3,  # -> 1024
        )
        self.layer4 = nn.Sequential(
            f.transition3,  # downsample
            f.denseblock4,  # -> 1024
            f.norm5,
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 2048, kernel_size=1, bias=False),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True),
        )


class TopoNet(nn.Module):
    def __init__(self, num_classes=2, attn_dropout=0.1, cls_dropout=0.1, num_heads=4,):
        super().__init__()

        #  DenseNet121 3D 
        self.img_densenet  = DenseNet3DEncoder(in_channels=1, out_channels=num_classes)
        self.topo_densenet = DenseNet3DEncoder(in_channels=1, out_channels=num_classes)
        
        #  BN -> GN
        replace_bn3d_with_gn(self.topo_densenet, prefer_groups=32)
            
        self.img_layers = nn.ModuleList([
            self.img_densenet.layer1,
            self.img_densenet.layer2,
            self.img_densenet.layer3,
            self.img_densenet.layer4,
        ])
        self.topo_layers = nn.ModuleList([
            self.topo_densenet.layer1,
            self.topo_densenet.layer2,
            self.topo_densenet.layer3,
            self.topo_densenet.layer4,
        ])
        
        # d_model [256, 384, 512, 768]  [512, 768, 768, 768] [192,192,384]
        # num_heads 4, 6, 8
        # target_spatial 10,8,6,4  12,10,8,6
        # stage3/4 stage2/3/4
        self.tam_modules = nn.ModuleList([
            None,
            CrossAttention(
                in_channels=512,
                d_model=128,
                n_heads=num_heads,
                dropout=attn_dropout,
                target_spatial=(6, 6, 6)
            ),
            CrossAttention(
                in_channels=1024,
                d_model=128,
                n_heads=num_heads,
                dropout=attn_dropout,
                target_spatial=(4, 4, 4)
            ),
            CrossAttention(
                in_channels=2048,
                d_model=256,
                n_heads=num_heads,
                dropout=attn_dropout,
                target_spatial=(3,3,3)
            ),
        ])
        # stage2: x_img(24^3) -> Q:4^3=64; ROI -> KV:6^3=216
        self.tam_modules[1].target_spatial_q  = (6, 6, 6)
        self.tam_modules[1].target_spatial_kv = (6,6,6)
        self.tam_modules[1].kv_topk_ratio = 0.6
        self.tam_modules[1].kv_min_tokens = 32
        self.tam_modules[1].kv_score_mode = "l2"

        # stage3: x_img(12^3) -> Q:3^3=27; ROI -> KV:4^3=64
        self.tam_modules[2].target_spatial_q  = (4,4,4)
        self.tam_modules[2].target_spatial_kv = (4,4,4)
        self.tam_modules[2].kv_topk_ratio = 0.5
        self.tam_modules[2].kv_min_tokens = 16
        self.tam_modules[2].kv_score_mode = "l2"

        # stage4: x_img(6^3) -> Q:2^3=8; ROI -> KV:3^3=27
        self.tam_modules[3].target_spatial_q  = (3,3,3)
        self.tam_modules[3].target_spatial_kv = (3,3,3)
        self.tam_modules[3].kv_topk_ratio = 0.5     
        self.tam_modules[3].kv_min_tokens = 8
        self.tam_modules[3].kv_score_mode = "l2"
        
        # enhance block
        self.img_enhance = nn.ModuleList([DWSpatialGate3D(c) for c in [256, 512,1024,2048]]) 
        self.topo_enhance = nn.ModuleList([PartialDWRes3D(c) for c in [256, 512,1024,2048]])    
       
            
        # gate
        self.fusion_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(4)
        ])
        
        # cls（2048 -> 256 -> num_classes）
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # share
        self.cls_attn_pool = AttnPool3d(in_ch=2048)  # share
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256), 
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(256, num_classes)
        )
            
        self.topo_attn_pool = AttnPool3d(in_ch=2048)  # without share
        # self.aux_head_topo = nn.Linear(2048, num_classes)
        self.aux_head_topo = nn.Sequential(
            nn.Linear(2048, 256), 
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(256, num_classes)
        )

    @staticmethod
    def _forward_stem(net, x):     
        if hasattr(net, "stem"):
            return net.stem(x)

        x = net.conv1(x)
        x = net.bn1(x)
        
        if hasattr(net, "act"):
            x = net.act(x)
        elif hasattr(net, "relu"):
            x = net.relu(x)
        else:
            x = F.relu(x, inplace=True)

        if hasattr(net, "no_max_pool") and net.no_max_pool:
            return x

        if hasattr(net, "maxpool"):
            x = net.maxpool(x)

        return x

    def forward(self, img, topo, return_aux=False):
        # Initial stem
        x_img  = self._forward_stem(self.img_densenet, img)  
        x_topo = self._forward_stem(self.topo_densenet, topo)

        for i in range(4):
            # stage-wise feature extraction
            x_img  = self.img_layers[i](x_img)     # img encoder
            x_topo = self.topo_layers[i](x_topo)   # topo encoder
            
            if i > 1:
                x_img_raw = x_img
            
                # lightweight attention on img branch, topo branch
                x_img = self.img_enhance[i](x_img)
                x_topo = self.topo_enhance[i](x_topo)
                
                # topo-guided modulation (incremental delta) for image features
                x_fuse = self.tam_modules[i](x_img, x_topo)      
                # x_fuse = x_img + x_topo
                
                # learnable scalar gate controls fusion strength per stage
                g = torch.sigmoid(self.fusion_gates[i])
                x_img = x_img_raw + g * x_fuse
                # x_img = (1.0 - g) * x_img_raw + g * x_fuse
 
        # out = self.global_pool(x_img)  # [B, 2048, 1, 1, 1]
        out = self.cls_attn_pool(x_img)
        
        out = out.view(out.size(0), -1)
        out = self.classifier(out)     # [B, num_classes]
        
        if not return_aux:
            return out
        
        # optional auxiliary head on topo branch for deep supervision
        feat_topo = self.topo_attn_pool(x_topo)                  # [B, 2048, 1, 1, 1]
        feat_topo_flat = feat_topo.view(feat_topo.size(0), -1)   
        aux_topo_logits = self.aux_head_topo(feat_topo_flat)  # [B, num_classes]

        return out, aux_topo_logits


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TopoNet(num_classes=2).to(device)
    summary(
        model,
        input_size=[(2, 1, 96, 96, 96),   # img
                    (2, 1, 96, 96, 96)],  # topo
        col_names=("input_size", "output_size", "num_params"),
        depth=3,
    )
