import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# img block
class ECA3D(nn.Module):
    """Efficient Channel Attention for 3D: very lightweight."""
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # x: (B,C,D,H,W)
        y = self.pool(x).flatten(2)         # (B,C,1) -> flatten -> (B,C,1)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)  # (B,1,C) -> (B,1,C) -> back
        y = torch.sigmoid(y).view(x.size(0), x.size(1), 1, 1, 1)
        return x * y


class SimAM3D(nn.Module):
    def __init__(self, lam=1e-4):
        super().__init__()
        self.lam = lam
    def forward(self, x):
        # x: (B,C,D,H,W)
        mu = x.mean(dim=(2,3,4), keepdim=True)
        var = ((x - mu)**2).mean(dim=(2,3,4), keepdim=True)
        e = (x - mu).pow(2) / (4.0 * (var + self.lam)) + 0.5
        attn = torch.sigmoid(1.0 / e)
        return x * attn


class DWRes3D(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        p = k//2
        self.dw = nn.Conv3d(ch, ch, k, padding=p, groups=ch, bias=False)
        self.pw = nn.Conv3d(ch, ch, 1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)
        self.act = nn.SiLU(inplace=True)
        nn.init.zeros_(self.pw.weight)  
    def forward(self, x):
        y = self.act(self.gn(self.pw(self.dw(x))))
        return x + y
    

class DWSpatialGate3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv3d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv3d(c, 1, 1, bias=True)

    def forward(self, x):
        w = torch.sigmoid(self.pw(self.dw(x)))          # (B,1,D,H,W)
        return x * (1.0 + w)


class CBAM3D(nn.Module):
    def __init__(self, c, r=8, k_spatial=7):
        super().__init__()
        mid = max(4, c // r)
        self.mlp = nn.Sequential(
            nn.Conv3d(c, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, c, 1, bias=False),
        )
        self.spatial = nn.Conv3d(2, 1, kernel_size=k_spatial, padding=k_spatial//2, bias=False)

    def forward(self, x):
        # Channel attn
        avg = x.mean(dim=(2,3,4), keepdim=True)
        mx  = x.amax(dim=(2,3,4), keepdim=True)
        wc = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x  = x * (1.0 + wc)

        # Spatial attn
        avg2 = x.mean(dim=1, keepdim=True)
        mx2  = x.amax(dim=1, keepdim=True)
        ws = torch.sigmoid(self.spatial(torch.cat([avg2, mx2], dim=1)))
        return x * (1.0 + ws)
 
    
class PartialDWRes3D(nn.Module):
    def __init__(self, ch, k=3, dilation=1, eps=1e-6, gn_groups=32):
        super().__init__()
        p = (k // 2) * dilation
        self.dw = nn.Conv3d(ch, ch, k, padding=p, dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv3d(ch, ch, 1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(gn_groups, ch), num_channels=ch)
        self.act = nn.SiLU(inplace=True)
        self.eps = eps
        self.dilation = dilation
        self.p = p
        self.k = k
        # 用于 mask 计数的 ones kernel
        self.register_buffer("ones", torch.ones(1, 1, k, k, k))
        # 零初始化：初始≈恒等
        nn.init.zeros_(self.pw.weight)

    def forward(self, x):
        # x: (B,C,D,H,W), 稀疏时用激活区域当 mask
        m = (x.abs().sum(dim=1, keepdim=True) > 0).float()  # (B,1,D,H,W)
        xm = x * m

        y = self.dw(xm)
        den = F.conv3d(m, self.ones, padding=self.p, dilation=self.dilation)
        y = y / (den + self.eps)

        y = self.pw(y)
        y = self.act(self.gn(y))
        return x + y

    
class MaskedSE3D(nn.Module):
    def __init__(self, ch, r=8, eps=1e-6):
        super().__init__()
        hidden = max(ch // r, 8)
        self.fc1 = nn.Conv3d(ch, hidden, 1)
        self.fc2 = nn.Conv3d(hidden, ch, 1)
        self.eps = eps
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        m = (x.abs().sum(dim=1, keepdim=True) > 0).float()
        s = (x * m).sum(dim=(2,3,4), keepdim=True)
        c = m.sum(dim=(2,3,4), keepdim=True) + self.eps
        avg = s / c
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(avg), inplace=True)))
        return x * (1.0 + w)  # residual-style 更稳
  

class SparseGate(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.gate = nn.Conv3d(in_ch, 1, kernel_size=1, bias=True)
       
    def forward(self, x,):
        w = torch.sigmoid(self.gate(x))     # (B,1,D,H,W)
        y = x * w                           # (B,C,D,H,W)
        return y    
    
    
class AttnPool3d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.attn = nn.Conv3d(in_ch, 1, 1)
    def forward(self, x):
        B,C,D,H,W = x.shape
        a = torch.softmax(self.attn(x).view(B,1,-1), dim=-1)
        y = (x.view(B,C,-1) * a).sum(dim=-1).view(B,C,1,1,1)
        return y


# v1.0 dropout
class MultiHeadTopoAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, dropout=0.1, topo_pool_ratio=4):
        super().__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.dim_head = in_channels // num_heads
        self.topo_pool_ratio = topo_pool_ratio  

        self.query_proj = nn.Linear(in_channels, in_channels)
        self.key_proj   = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        self.out_proj   = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, topo_feat, img_feat):
        # Input: [B, C, D, H, W]
        B, C, Di, Hi, Wi = img_feat.shape

        #  tokens（N_img）
        N_img = Di * Hi * Wi
        img_flat = img_feat.view(B, C, N_img).transpose(1, 2)  # [B, N_img, C]

        #  N_topo << N_img
        Dt, Ht, Wt = topo_feat.shape[2:]
        td = max(1, Dt // self.topo_pool_ratio)
        th = max(1, Ht // self.topo_pool_ratio)
        tw = max(1, Wt // self.topo_pool_ratio)
        topo_ds = F.adaptive_avg_pool3d(topo_feat, (td, th, tw))  # [B, C, td, th, tw]

        # tokens（N_topo）
        N_topo = td * th * tw
        topo_flat = topo_ds.view(B, C, N_topo).transpose(1, 2)   # [B, N_topo, C]

        # Project Q/K/V（Q: img tokens, K/V: topo tokens）
        Q = self.query_proj(img_flat)   # [B, N_img,  C]
        K = self.key_proj(topo_flat)    # [B, N_topo, C]
        V = self.value_proj(topo_flat)  # [B, N_topo, C]

        # Split heads
        Q = Q.view(B, N_img,  self.num_heads, self.dim_head).transpose(1, 2)  # [B, heads, N_img,  dim]
        K = K.view(B, N_topo, self.num_heads, self.dim_head).transpose(1, 2)  # [B, heads, N_topo, dim]
        V = V.view(B, N_topo, self.num_heads, self.dim_head).transpose(1, 2)  # [B, heads, N_topo, dim]

        #  Cross-Attn: N_img × N_topo
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_head ** 0.5)  # [B, heads, N_img, N_topo]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(self.dropout(attn_weights), V)  # [B, heads, N_img, dim]

        # Concat heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_img, C)
        attn_out = self.out_proj(attn_out)

        # Residual + reshape
        out = img_flat + attn_out
        return out.transpose(1, 2).view(B, C, Di, Hi, Wi)
    
# basic
class CrossAttention(nn.Module):
    """
    x1, x2: (B, N, D)  
    return: (B, N, D) 
    """
    def __init__(self, in_channels, d_model=768, n_heads=4, d_ff=0, dropout=0.1, gate_mode='scalar', target_spatial=(6, 6, 6)):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.target_spatial = target_spatial
        
        # Pre-LN
        self.norm1_x1 = nn.LayerNorm(in_channels)
        self.norm1_x2 = nn.LayerNorm(in_channels)
        
        # x1<-x2: Q来自x1，K/V来自x2
        self.q1 = nn.Linear(in_channels, d_model, bias=False)
        self.k2 = nn.Linear(in_channels, d_model, bias=False)
        self.v2 = nn.Linear(in_channels, d_model, bias=False)

        # x2<-x1: Q来自x2，K/V来自x1
        self.q2 = nn.Linear(in_channels, d_model, bias=False)
        self.k1 = nn.Linear(in_channels, d_model, bias=False)
        self.v1 = nn.Linear(in_channels, d_model, bias=False)

        # 融合后输出投影
        self.out = nn.Linear(d_model, in_channels, bias=True)

        # 残差尺寸对齐（若你后续让 d_model != D，这里会用到）
        self.res_proj = nn.Identity()

        if gate_mode == 'scalar':
            self.gate = nn.Parameter(torch.zeros(1))            
        elif gate_mode == 'per_head':
            self.gate = nn.Parameter(torch.zeros(n_heads))        
        elif gate_mode == 'per_head_channel':
            self.gate = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        else:
            raise ValueError("gate_mode ∈ {'scalar','per_head','per_head_channel'}")

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        
        self.target_spatial_q = target_spatial
        self.target_spatial_kv = target_spatial
        self.kv_topk_ratio = 0.25
        self.kv_min_tokens = 32
        self.kv_score_mode = "l2"
        
        self.use_ffn = d_ff and d_ff > 0
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(in_channels)
            self.ffn = nn.Sequential(
                nn.Linear(in_channels, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, in_channels),
                nn.Dropout(dropout),
            )

    def _split_heads(self, x):
        # (B, N, D) -> (B, H, N, d_head)
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x):
        # (B, H, N, d_head) -> (B, N, D)
        B, H, N, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * Dh)

    def _sdpa(self, Q, K, V, mask=None):
        """
        Q,K,V: (B, H, N, d_head)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,N,N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, V)  # (B,H,N,d_head)
        return out
    
    def _to_sequence(self, x):
        """
        x: (B, C, H, W, Dv) -> seq: (B, N, C), shape
        """
        B, C, H, W, Dv = x.shape
        N = H * W * Dv
        seq = x.view(B, C, N).permute(0, 2, 1).contiguous()  # (B,N,C)
        spatial = (H, W, Dv)
        return seq, spatial

    def _from_sequence(self, y, spatial):
        """
        y: (B, N, C) -> (B, C, H, W, Dv)
        """
        B, N, C = y.shape
        H, W, Dv = spatial
        x = y.permute(0, 2, 1).contiguous().view(B, C, H, W, Dv)
        return x

    # basic
    # def forward(self, x1, x2, mask=None):
    #     """
    #     x1: 整图分支特征 (B,C,*,*,*)   —— Query 来源（被增强对象）
    #     x2: ROI 分支特征  (B,C,*,*,*)   —— Key/Value 来源（提供信息）
    #     return: (B,C,*,*,*) 与 x1 同形状
    #     """
    #     orig_spatial = x1.shape[2:]  # 只对齐回 x1 的空间

    #     # =========================
    #     # 1) 不同分辨率下采样：允许 N1 != N2
    #     # =========================
    #     ts_q  = getattr(self, "target_spatial_q",  getattr(self, "target_spatial", (6,6,6)))
    #     ts_kv = getattr(self, "target_spatial_kv", getattr(self, "target_spatial", (6,6,6)))

    #     # x1(整图)：avg 即可（更稳）
    #     x1_ds = F.adaptive_avg_pool3d(x1, ts_q)

    #     # x2(ROI)：avg+max 更不容易把小病灶平均没
    #     # x2_ds = F.adaptive_max_pool3d(x2,ts_kv)
    #     x2_avg = F.adaptive_avg_pool3d(x2, ts_kv)
    #     x2_max = F.adaptive_max_pool3d(x2, ts_kv)
    #     x2_ds  = 0.5 * (x2_avg + x2_max)

    #     s1, spatial1 = self._to_sequence(x1_ds)  # (B,N1,C)
    #     s2, _        = self._to_sequence(x2_ds)  # (B,N2,C)

    #     B, N1, C = s1.shape
    #     _, N2, _ = s2.shape

    #     # Pre-LN
    #     s1n = self.norm1_x1(s1)
    #     s2n = self.norm1_x2(s2)

    #     # =========================
    #     # 2) 单向交叉注意力：x1 <- x2
    #     # =========================
    #     Q1 = self._split_heads(self.q1(s1n))  # (B,H,N1,d)

    #     # =========================
    #     # 3) ROI Top-K token pruning：只保留最“强”的少量 K/V（关键优化）
    #     # =========================
    #     kv_topk_ratio = getattr(self, "kv_topk_ratio", 0.25)  # 默认保留 25%
    #     kv_min_tokens = getattr(self, "kv_min_tokens", 32)    # 默认最少 32 个
    #     kv_score_mode = getattr(self, "kv_score_mode", "l2")  # "l2" 或 "absmean"

    #     k = max(kv_min_tokens, int(kv_topk_ratio * N2))
    #     k = min(k, N2)

    #     if kv_score_mode == "l2":
    #         score = torch.norm(s2n, dim=-1)           # (B,N2)
    #     else:
    #         score = s2n.abs().mean(dim=-1)            # (B,N2)

    #     idx = score.topk(k, dim=1, largest=True).indices  # (B,k)

    #     # 先投影到 d_model，再按 idx gather（最直观、改动最少）
    #     K2 = self.k2(s2n)  # (B,N2,d_model)
    #     V2 = self.v2(s2n)  # (B,N2,d_model)

    #     idx_k = idx.unsqueeze(-1).expand(-1, -1, self.d_model)  # (B,k,d_model)
    #     K2 = K2.gather(1, idx_k)  # (B,k,d_model)
    #     V2 = V2.gather(1, idx_k)  # (B,k,d_model)

    #     K2 = self._split_heads(K2)  # (B,H,k,d)
    #     V2 = self._split_heads(V2)  # (B,H,k,d)

    #     # 如果你传了 mask（形状建议 Bx1xN1xN2），这里同步裁剪到 Bx1xN1xk
    #     if mask is not None:
    #         if mask.shape[0] == 1 and B > 1:
    #             mask = mask.expand(B, -1, -1, -1)
    #         if mask.shape[-2] == N1 and mask.shape[-1] == N2:
    #             idx_m = idx.unsqueeze(1).unsqueeze(2).expand(B, 1, N1, k)  # (B,1,N1,k)
    #             mask = mask.gather(-1, idx_m)

    #     # =========================
    #     # 4) 注意力 + 输出投影（只用 O1）
    #     # =========================
    #     # scores: (B,H,N1,k)
    #     scores = torch.matmul(Q1, K2.transpose(-2, -1)) / math.sqrt(self.d_head)
    #     if mask is not None:
    #         scores = scores.masked_fill(~mask, float("-inf"))
    #     attn = F.softmax(scores, dim=-1)
    #     attn = self.attn_drop(attn)
    #     O1 = torch.matmul(attn, V2)  # (B,H,N1,d)

    #     O = self._merge_heads(O1)             # (B,N1,d_model)
    #     O = self.proj_drop(self.out(O))       # (B,N1,C)

    #     # 残差：只用 x1（更符合 ROI 引导整图）
    #     y = O + self.res_proj(s1)

    #     if self.use_ffn:
    #         y = y + self.ffn(self.norm2(y))

    #     y = self._from_sequence(y, spatial1)  # (B,C,ts_q)
    #     y = F.interpolate(y, size=orig_spatial, mode="trilinear", align_corners=False)
    #     return y
        
    # [Variant B] I -> T
    # def forward(self, x1, x2, mask=None):
    #     """
    #     [Variant B] I -> T
    #     Target (Query): x2 (Topology)
    #     Source (KV):    x1 (Image) -> NO Pruning (Keep Dense Context)
    #     """
    #     orig_spatial = x2.shape[2:] 

    #     ts_q  = getattr(self, "target_spatial_q",  (6,6,6))
    #     ts_kv = getattr(self, "target_spatial_kv", (6,6,6))

    #     # 1. 下采样
    #     # x2 (Topology Query): 混合池化保留细节
    #     x2_avg = F.adaptive_avg_pool3d(x2, ts_q)
    #     x2_max = F.adaptive_max_pool3d(x2, ts_q)
    #     x2_ds  = 0.5 * (x2_avg + x2_max)

    #     # x1 (Image KV): AvgPool
    #     x1_ds = F.adaptive_avg_pool3d(x1, ts_kv) 

    #     s2, spatial2 = self._to_sequence(x2_ds) # Query (B, N2, C)
    #     s1, _        = self._to_sequence(x1_ds) # KV    (B, N1, C)

    #     s2n = self.norm1_x2(s2)
    #     s1n = self.norm1_x1(s1)

    #     # 2. Attention Prep
    #     # Q 来自 x2
    #     Q2 = self._split_heads(self.q2(s2n)) # (B, H, N2, d)

    #     # === [Modification] Image 分支不进行 Pruning ===
    #     # K, V 来自 x1 (全量)
    #     K1 = self._split_heads(self.k1(s1n)) # (B, H, N1, d)
    #     V1 = self._split_heads(self.v1(s1n)) # (B, H, N1, d)
        
    #     # 3. 计算 Attention (Full Attention)
    #     # scores: (B, H, N2, N1)
    #     scores = torch.matmul(Q2, K1.transpose(-2, -1)) / math.sqrt(self.d_head)
    #     attn = F.softmax(scores, dim=-1)
    #     attn = self.attn_drop(attn)
    #     O2 = torch.matmul(attn, V1)

    #     # 4. 输出 & 残差
    #     O = self._merge_heads(O2)
    #     O = self.proj_drop(self.out(O))
        
    #     y = O + self.res_proj(s2)

    #     if self.use_ffn:
    #         y = y + self.ffn(self.norm2(y))

    #     y = self._from_sequence(y, spatial2)
    #     y = F.interpolate(y, size=orig_spatial, mode="trilinear", align_corners=False)
    #     return y

    # [Variant C] Parallel
    # def forward(self, x1, x2, mask=None):
    #     """
    #     [Variant C] Parallel
    #     Stream A (T->I): Prune T (Safe Fix)
    #     Stream B (I->T): Full I
    #     """
    #     ts = getattr(self, "target_spatial", (6,6,6))
        
    #     # 下采样
    #     x1_ds = F.adaptive_avg_pool3d(x1, ts) # Image
    #     x2_avg = F.adaptive_avg_pool3d(x2, ts)
    #     x2_max = F.adaptive_max_pool3d(x2, ts)
    #     x2_ds = 0.5 * (x2_avg + x2_max)       # Topology

    #     s1, sp1 = self._to_sequence(x1_ds)
    #     s2, sp2 = self._to_sequence(x2_ds)
        
    #     s1n, s2n = self.norm1_x1(s1), self.norm1_x2(s2)
    #     B, N1, _ = s1.shape
    #     B, N2, _ = s2.shape

    #     # ==========================================
    #     # Stream A: Enhance Image (KV = Topology -> PRUNE)
    #     # ==========================================
    #     Q1 = self._split_heads(self.q1(s1n))
        
    #     # --- [Fixed] Safe Pruning Logic ---
    #     kv_topk_ratio = getattr(self, "kv_topk_ratio", 0.25)
    #     kv_min_tokens = getattr(self, "kv_min_tokens", 32)
    #     kv_score_mode = getattr(self, "kv_score_mode", "l2")

    #     # 1. 计算期望 k
    #     k = int(kv_topk_ratio * N2)
    #     k = max(k, kv_min_tokens) # 保底

    #     # 2. [关键修复] 封顶 k，防止 k > N2 导致 index out of range
    #     k = min(k, N2)

    #     # 3. 计算分数
    #     if kv_score_mode == "l2":
    #         score_T = torch.norm(s2n, dim=-1)
    #     else:
    #         score_T = s2n.abs().mean(dim=-1)
            
    #     idx_T = score_T.topk(k, dim=1, largest=True).indices
    #     # ----------------------------------
        
    #     K2 = self.k2(s2n)
    #     V2 = self.v2(s2n)
        
    #     # Gather Top-K Topology tokens
    #     idx_k = idx_T.unsqueeze(-1).expand(-1, -1, self.d_model)
    #     K2_pruned = K2.gather(1, idx_k)
    #     V2_pruned = V2.gather(1, idx_k)
        
    #     K2_p = self._split_heads(K2_pruned)
    #     V2_p = self._split_heads(V2_pruned)
        
    #     scores_A = torch.matmul(Q1, K2_p.transpose(-2,-1)) / math.sqrt(self.d_head)
    #     attn_A = self.attn_drop(F.softmax(scores_A, dim=-1))
    #     out_A = self._merge_heads(torch.matmul(attn_A, V2_p))
    #     res_A = self.proj_drop(self.out(out_A)) + s1

    #     # ==========================================
    #     # Stream B: Enhance Topology (KV = Image -> FULL)
    #     # ==========================================
    #     Q2 = self._split_heads(self.q2(s2n))
        
    #     # No Pruning for Image (s1)
    #     K1_full = self._split_heads(self.k1(s1n))
    #     V1_full = self._split_heads(self.v1(s1n))
        
    #     scores_B = torch.matmul(Q2, K1_full.transpose(-2,-1)) / math.sqrt(self.d_head)
    #     attn_B = self.attn_drop(F.softmax(scores_B, dim=-1))
    #     out_B = self._merge_heads(torch.matmul(attn_B, V1_full))
    #     res_B = self.proj_drop(self.out(out_B)) + s2

    #     # ==========================================
    #     # Fusion
    #     # ==========================================
    #     y_I = self._from_sequence(res_A, sp1)
    #     y_T = self._from_sequence(res_B, sp2)
        
    #     orig_spatial = x1.shape[2:]
    #     y_I_up = F.interpolate(y_I, size=orig_spatial, mode="trilinear", align_corners=False)
    #     y_T_up = F.interpolate(y_T, size=orig_spatial, mode="trilinear", align_corners=False)
        
    #     return y_I_up + y_T_up
        
    # [Variant D] I -> T -> I
    def forward(self, x1, x2, mask=None):
        """
        [Variant D] I -> T -> I
        Step 1: KV=I (Full)
        Step 2: KV=T (Pruned & Safe Fix)
        """
        orig_spatial = x1.shape[2:]
        ts = getattr(self, "target_spatial", (6,6,6))

        x1_ds = F.adaptive_avg_pool3d(x1, ts)
        x2_ds = 0.5 * (F.adaptive_avg_pool3d(x2, ts) + F.adaptive_max_pool3d(x2, ts))

        s1, sp1 = self._to_sequence(x1_ds)
        s2, sp2 = self._to_sequence(x2_ds)
        B, N2, _ = s2.shape

        s1n = self.norm1_x1(s1)
        s2n = self.norm1_x2(s2)

        # ==========================================
        # Step 1: I -> T (Update T using Full Image)
        # ==========================================
        Q2 = self._split_heads(self.q2(s2n))
        K1_full = self._split_heads(self.k1(s1n))
        V1_full = self._split_heads(self.v1(s1n))
        
        scores_1 = torch.matmul(Q2, K1_full.transpose(-2,-1)) / math.sqrt(self.d_head)
        attn_1 = F.softmax(scores_1, dim=-1)
        out_T = self._merge_heads(torch.matmul(attn_1, V1_full))
        
        s2_updated = self.proj_drop(self.out(out_T)) + s2 

        # ==========================================
        # Step 2: T' -> I (Update I using Pruned New T)
        # ==========================================
        s2_updated_n = self.norm1_x2(s2_updated) # Re-norm
        
        Q1 = self._split_heads(self.q1(s1n))
        
        # --- [Fixed] Safe Pruning Logic (on Updated Topology) ---
        kv_topk_ratio = getattr(self, "kv_topk_ratio", 0.25)
        kv_min_tokens = getattr(self, "kv_min_tokens", 32)
        kv_score_mode = getattr(self, "kv_score_mode", "l2")
        
        # 1. 计算期望 k
        k = int(kv_topk_ratio * N2)
        k = max(k, kv_min_tokens)
        
        # 2. [关键修复] 封顶 k
        k = min(k, N2)
        
        # 3. 计算分数 (基于更新后的特征)
        if kv_score_mode == "l2":
            score_T_new = torch.norm(s2_updated_n, dim=-1)
        else:
            score_T_new = s2_updated_n.abs().mean(dim=-1)
            
        idx_T = score_T_new.topk(k, dim=1, largest=True).indices
        # -----------------------------------------------------
        
        # 从更新后的 T 中采样
        K2_new = self.k2(s2_updated_n)
        V2_new = self.v2(s2_updated_n)
        
        idx_k = idx_T.unsqueeze(-1).expand(-1, -1, self.d_model)
        K2_pruned = K2_new.gather(1, idx_k)
        V2_pruned = V2_new.gather(1, idx_k)
        
        K2_p = self._split_heads(K2_pruned)
        V2_p = self._split_heads(V2_pruned)

        scores_2 = torch.matmul(Q1, K2_p.transpose(-2,-1)) / math.sqrt(self.d_head)
        attn_2 = self.attn_drop(F.softmax(scores_2, dim=-1))
        out_I = self._merge_heads(torch.matmul(attn_2, V2_p))

        # Final
        y = self.proj_drop(self.out(out_I)) + s1
        
        if self.use_ffn:
            y = y + self.ffn(self.norm2(y))

        y = self._from_sequence(y, sp1)
        y = F.interpolate(y, size=orig_spatial, mode="trilinear", align_corners=False)
        return y
    
# DualImageNet
# class CrossAttention(nn.Module):
#     """
#     x1, x2: (B, N, D)  
#     return: (B, N, D) 
#     """
#     def __init__(self, in_channels, d_model=768, n_heads=4, d_ff=0, dropout=0.1, gate_mode='scalar', target_spatial=(6, 6, 6)):
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
#         self.in_channels = in_channels
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_head = d_model // n_heads
#         self.target_spatial = target_spatial
        
#         # Pre-LN
#         self.norm1_x1 = nn.LayerNorm(in_channels)
#         self.norm1_x2 = nn.LayerNorm(in_channels)
        
#         # x1<-x2: Q来自x1，K/V来自x2
#         self.q1 = nn.Linear(in_channels, d_model, bias=False)
#         self.k2 = nn.Linear(in_channels, d_model, bias=False)
#         self.v2 = nn.Linear(in_channels, d_model, bias=False)

#         # ... (q2, k1, v1 定义省略，forward中未使用，为保持最少修改不删除) ...
#         self.q2 = nn.Linear(in_channels, d_model, bias=False)
#         self.k1 = nn.Linear(in_channels, d_model, bias=False)
#         self.v1 = nn.Linear(in_channels, d_model, bias=False)

#         # 融合后输出投影
#         self.out = nn.Linear(d_model, in_channels, bias=True)
#         self.res_proj = nn.Identity()

#         if gate_mode == 'scalar':
#             self.gate = nn.Parameter(torch.zeros(1))            
#         elif gate_mode == 'per_head':
#             self.gate = nn.Parameter(torch.zeros(n_heads))        
#         elif gate_mode == 'per_head_channel':
#             self.gate = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
#         else:
#             raise ValueError("gate_mode ∈ {'scalar','per_head','per_head_channel'}")

#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_drop = nn.Dropout(dropout)
        
#         self.target_spatial_q = target_spatial
#         self.target_spatial_kv = target_spatial
        
#         # ### <<< 修改 1: 将 Top-K 比例改为 1.0 (保留所有 token) >>>
#         # 原因：img 分支是稠密的，不像 Topo 分支那样稀疏。
#         # 强制保留所有特征，避免丢失图像纹理信息。
#         self.kv_topk_ratio = 1.0  # 原为 0.25
        
#         self.kv_min_tokens = 32
#         self.kv_score_mode = "l2"
        
#         self.use_ffn = d_ff and d_ff > 0
#         if self.use_ffn:
#             self.norm2 = nn.LayerNorm(in_channels)
#             self.ffn = nn.Sequential(
#                 nn.Linear(in_channels, d_ff),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(d_ff, in_channels),
#                 nn.Dropout(dropout),
#             )

#     def _split_heads(self, x):
#         B, N, _ = x.shape
#         return x.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3).contiguous()

#     def _merge_heads(self, x):
#         B, H, N, Dh = x.shape
#         return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * Dh)
    
#     def _to_sequence(self, x):
#         B, C, H, W, Dv = x.shape
#         N = H * W * Dv
#         seq = x.view(B, C, N).permute(0, 2, 1).contiguous() 
#         spatial = (H, W, Dv)
#         return seq, spatial

#     def _from_sequence(self, y, spatial):
#         B, N, C = y.shape
#         H, W, Dv = spatial
#         x = y.permute(0, 2, 1).contiguous().view(B, C, H, W, Dv)
#         return x

#     def forward(self, x1, x2, mask=None):
#         """
#         x1: img特征 (B,C,*,*,*) —— Query
#         x2: img特征 (B,C,*,*,*) —— Key/Value (不再是 ROI/Topo)
#         """
#         orig_spatial = x1.shape[2:] 

#         # 1) 不同分辨率下采样
#         ts_q  = getattr(self, "target_spatial_q",  getattr(self, "target_spatial", (6,6,6)))
#         ts_kv = getattr(self, "target_spatial_kv", getattr(self, "target_spatial", (6,6,6)))

#         x1_ds = F.adaptive_avg_pool3d(x1, ts_q)

#         # ### <<< 修改 2: x2 下采样方式统一为 AvgPool >>>
#         # 原因：原代码的 MaxPool 是为了突出稀疏的拓扑骨架。
#         # 对于 img/img，两路都是稠密图像，应使用一致的 AvgPool 以避免引入高频噪声或伪影。
#         # x2_ds = 0.5 * (F.adaptive_avg_pool3d(x2, ts_kv) + F.adaptive_max_pool3d(x2, ts_kv)) <-- 删除
#         x2_ds = F.adaptive_avg_pool3d(x2, ts_kv) # <-- 新增

#         s1, spatial1 = self._to_sequence(x1_ds) 
#         s2, _        = self._to_sequence(x2_ds) 

#         B, N1, C = s1.shape
#         _, N2, _ = s2.shape

#         s1n = self.norm1_x1(s1)
#         s2n = self.norm1_x2(s2)

#         # 2) 单向交叉注意力：x1 <- x2
#         Q1 = self._split_heads(self.q1(s1n)) 

#         # 3) ROI Top-K token pruning 逻辑优化
#         kv_topk_ratio = getattr(self, "kv_topk_ratio", 1.0) # 默认为 1.0
#         kv_min_tokens = getattr(self, "kv_min_tokens", 32)
        
#         k = max(kv_min_tokens, int(kv_topk_ratio * N2))
#         k = min(k, N2)

#         # ### <<< 修改 3: 性能优化 - 如果 k等于N2 (即全保留)，跳过 Gather >>>
#         # 原因：img/img 模式下不需要剪枝。直接计算 Norm、排序、Gather 非常耗时且无意义。
#         # 加入此判断可以显著加速训练。
#         if k < N2:
#             # --- 原 Top-K 逻辑 (仅当 kv_topk_ratio < 1.0 时执行) ---
#             kv_score_mode = getattr(self, "kv_score_mode", "l2")
#             if kv_score_mode == "l2":
#                 score = torch.norm(s2n, dim=-1)
#             else:
#                 score = s2n.abs().mean(dim=-1)
#             idx = score.topk(k, dim=1, largest=True).indices 

#             K2 = self.k2(s2n) 
#             V2 = self.v2(s2n) 
#             idx_k = idx.unsqueeze(-1).expand(-1, -1, self.d_model) 
#             K2 = K2.gather(1, idx_k) 
#             V2 = V2.gather(1, idx_k) 
            
#             # Mask 处理 (略，仅为了保持结构完整)
#             if mask is not None and mask.shape[-1] == N2:
#                  # ... (省略 mask gather 代码以简化) ...
#                  pass
#         else:
#             # --- 全量保留路径 (img/img 走这里) ---
#             K2 = self.k2(s2n)
#             V2 = self.v2(s2n)

#         K2 = self._split_heads(K2) 
#         V2 = self._split_heads(V2) 

#         # 4) 注意力 + 输出
#         scores = torch.matmul(Q1, K2.transpose(-2, -1)) / math.sqrt(self.d_head)
#         if mask is not None:
#              # 注意：如果 k < N2，mask 需要在上面被裁剪过；如果 k==N2，mask 直接用
#              if scores.shape[-1] == mask.shape[-1]:
#                 scores = scores.masked_fill(~mask, float("-inf"))
                
#         attn = F.softmax(scores, dim=-1)
#         attn = self.attn_drop(attn)
#         O1 = torch.matmul(attn, V2) 

#         O = self._merge_heads(O1) 
#         O = self.proj_drop(self.out(O)) 

#         y = O + self.res_proj(s1)

#         if self.use_ffn:
#             y = y + self.ffn(self.norm2(y))

#         y = self._from_sequence(y, spatial1) 
#         y = F.interpolate(y, size=orig_spatial, mode="trilinear", align_corners=False)
#         return y        


# DualTopoNet
# class CrossAttention(nn.Module):
#     """
#     针对 Topo/Topo 优化的 CrossAttention
#     x1, x2: 均为稀疏拓扑特征
#     """
#     def __init__(self, in_channels, d_model=768, n_heads=4, d_ff=0, dropout=0.1, gate_mode='scalar', target_spatial=(6, 6, 6)):
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
#         self.in_channels = in_channels
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_head = d_model // n_heads
#         self.target_spatial = target_spatial
        
#         # Pre-LN
#         self.norm1_x1 = nn.LayerNorm(in_channels)
#         self.norm1_x2 = nn.LayerNorm(in_channels)
        
#         # x1<-x2
#         self.q1 = nn.Linear(in_channels, d_model, bias=False)
#         self.k2 = nn.Linear(in_channels, d_model, bias=False)
#         self.v2 = nn.Linear(in_channels, d_model, bias=False)
        
#         # unused (kept for compatibility)
#         self.q2 = nn.Linear(in_channels, d_model, bias=False)
#         self.k1 = nn.Linear(in_channels, d_model, bias=False)
#         self.v1 = nn.Linear(in_channels, d_model, bias=False)

#         self.out = nn.Linear(d_model, in_channels, bias=True)
#         self.res_proj = nn.Identity()

#         if gate_mode == 'scalar':
#             self.gate = nn.Parameter(torch.zeros(1))            
#         elif gate_mode == 'per_head':
#             self.gate = nn.Parameter(torch.zeros(n_heads))        
#         elif gate_mode == 'per_head_channel':
#             self.gate = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
#         else:
#             raise ValueError("gate_mode error")

#         self.attn_drop = nn.Dropout(dropout)
#         self.proj_drop = nn.Dropout(dropout)
        
#         self.target_spatial_q = target_spatial
#         self.target_spatial_kv = target_spatial
        
#         # ### <<< 修改 1: 确保 Top-K 比例较低 (0.25 或更低) >>>
#         # 原因：Topo 数据极度稀疏，大部分区域是无效背景。
#         # 我们只需要关注那些“有值”的 token，过滤掉背景噪声。
#         self.kv_topk_ratio = 0.25 
        
#         self.kv_min_tokens = 32
#         self.kv_score_mode = "l2"
        
#         self.use_ffn = d_ff and d_ff > 0
#         if self.use_ffn:
#             self.norm2 = nn.LayerNorm(in_channels)
#             self.ffn = nn.Sequential(
#                 nn.Linear(in_channels, d_ff),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(d_ff, in_channels),
#                 nn.Dropout(dropout),
#             )

#     def _split_heads(self, x):
#         B, N, _ = x.shape
#         return x.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3).contiguous()

#     def _merge_heads(self, x):
#         B, H, N, Dh = x.shape
#         return x.permute(0, 2, 1, 3).contiguous().view(B, N, H * Dh)
    
#     def _to_sequence(self, x):
#         B, C, H, W, Dv = x.shape
#         N = H * W * Dv
#         seq = x.view(B, C, N).permute(0, 2, 1).contiguous() 
#         spatial = (H, W, Dv)
#         return seq, spatial

#     def _from_sequence(self, y, spatial):
#         B, N, C = y.shape
#         H, W, Dv = spatial
#         x = y.permute(0, 2, 1).contiguous().view(B, C, H, W, Dv)
#         return x

#     def forward(self, x1, x2, mask=None):
#         """
#         x1: Topo 特征 —— Query
#         x2: Topo 特征 —— Key/Value
#         """
#         orig_spatial = x1.shape[2:] 

#         ts_q  = getattr(self, "target_spatial_q",  getattr(self, "target_spatial", (6,6,6)))
#         ts_kv = getattr(self, "target_spatial_kv", getattr(self, "target_spatial", (6,6,6)))

#         # ### <<< 修改 2: Query (x1) 下采样改为 MaxPool >>>
#         # 原因：x1 现在也是稀疏拓扑。原代码的 AvgPool 会导致细微骨架模糊。
#         # MaxPool 能保留最强的结构特征作为查询依据。
#         # x1_ds = F.adaptive_avg_pool3d(x1, ts_q) <-- 删除
#         x1_avg = F.adaptive_avg_pool3d(x1, ts_q)
#         x1_max = F.adaptive_max_pool3d(x1, ts_q)
#         x1_ds  = 0.5 * (x1_avg + x1_max)

#         # ### <<< 修改 3: Key/Value (x2) 下采样改为纯 MaxPool >>>
#         # 原因：x2 是提供信息的源头。对于骨架/血管，AvgPool 会稀释信号。
#         # 纯 MaxPool 是处理稀疏二值/灰度结构特征的标准做法。
#         # x2_ds = 0.5 * (avg + max) <-- 删除
#         x2_avg = F.adaptive_avg_pool3d(x2, ts_kv)
#         x2_max = F.adaptive_max_pool3d(x2, ts_kv)
#         x2_ds  = 0.5 * (x2_avg + x2_max)

#         s1, spatial1 = self._to_sequence(x1_ds) 
#         s2, _        = self._to_sequence(x2_ds) 

#         B, N1, C = s1.shape
#         _, N2, _ = s2.shape

#         s1n = self.norm1_x1(s1)
#         s2n = self.norm1_x2(s2)

#         Q1 = self._split_heads(self.q1(s1n)) 

#         # --- Top-K Token Pruning (保留) ---
#         # 这里的逻辑不需要修改，因为我们希望在 Topo/Topo 中使用 Pruning。
#         # 只要保证 init 里 kv_topk_ratio < 1.0 即可。
#         kv_topk_ratio = getattr(self, "kv_topk_ratio", 0.25)
#         kv_min_tokens = getattr(self, "kv_min_tokens", 32)
#         kv_score_mode = getattr(self, "kv_score_mode", "l2")

#         k = max(kv_min_tokens, int(kv_topk_ratio * N2))
#         k = min(k, N2)

#         if kv_score_mode == "l2":
#             score = torch.norm(s2n, dim=-1)
#         else:
#             score = s2n.abs().mean(dim=-1)
        
#         idx = score.topk(k, dim=1, largest=True).indices 

#         K2 = self.k2(s2n) 
#         V2 = self.v2(s2n) 
#         idx_k = idx.unsqueeze(-1).expand(-1, -1, self.d_model) 
#         K2 = K2.gather(1, idx_k) 
#         V2 = V2.gather(1, idx_k) 

#         K2 = self._split_heads(K2) 
#         V2 = self._split_heads(V2) 

#         # Attention + Output
#         scores = torch.matmul(Q1, K2.transpose(-2, -1)) / math.sqrt(self.d_head)
        
#         # Mask 处理
#         if mask is not None and mask.shape[-1] == N2:
#              # (这里省略 mask gather 代码，假设 mask 也会被相应处理)
#              pass
                
#         attn = F.softmax(scores, dim=-1)
#         attn = self.attn_drop(attn)
#         O1 = torch.matmul(attn, V2) 

#         O = self._merge_heads(O1) 
#         O = self.proj_drop(self.out(O)) 

#         y = O + self.res_proj(s1)

#         if self.use_ffn:
#             y = y + self.ffn(self.norm2(y))

#         y = self._from_sequence(y, spatial1) 
#         y = F.interpolate(y, size=orig_spatial, mode="trilinear", align_corners=False)
#         return y

