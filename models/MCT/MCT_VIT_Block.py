import torch
import torch.nn as nn
import torch.nn.functional as F



class MCT_ViT_Block(nn.Module):
    def __init__(self, token_dim = 256, num_heads=8): # 멀티헤드 8개로 나눔
        super(MCT_ViT_Block, self).__init__()
        # Linear layers to match dimensions for concatenation
        self.cls_linear = nn.Linear(token_dim, token_dim)
        self.token_linear = nn.Linear(token_dim, token_dim)
        
        # Multiscale Conv
        self.conv3 = nn.Conv1d(token_dim, token_dim, kernel_size=3, padding=1) # 동일한 출력을 위해 padding 조절
        self.conv5 = nn.Conv1d(token_dim, token_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(token_dim, token_dim, kernel_size=7, padding=3)

        # Feed-forward(1)
        self.norm1 = nn.LayerNorm(token_dim)
        self.conv1x1 = nn.Conv1d(token_dim * 3, token_dim, kernel_size=1)  # 차원을 맞추기 위한 1x1 컨볼루션
        self.gelu = nn.GELU()

        # Feed-forward(2)-(MHSA)
        self.norm2 = nn.LayerNorm(token_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim = token_dim, num_heads = 8, batch_first = True)

    def forward(self, cls_token, ecg_tokens, pcg_tokens):
        '''
        input
        cls_token_shape: torch.Size([128, 256, 1])
        ecg_tokens_shape: torch.Size([128, 313, 256]), pcg_tokens_shape: torch.Size([128, 313, 256])
        '''
        cls_token = cls_token.transpose(1, 2)  # (batch_size, 1, token_dim) # 차원 맞춰주기

        # Linear convolution 
        cls_token_proj = self.cls_linear(cls_token)  # (batch_size, 1, token_dim)
        ecg_tokens_proj = self.token_linear(ecg_tokens)  # (batch_size, L, token_dim)
        pcg_tokens_proj = self.token_linear(pcg_tokens)  # (batch_size, L, token_dim)

        # concat(Cls token 벡터를 시퀀스의 길이만큼 repeat시킴)
        ecg_concat = torch.cat([cls_token_proj.expand(-1, ecg_tokens_proj.size(1), -1), ecg_tokens_proj], dim=-1)  # (batch_size, L, token_dim * 2)
        pcg_concat = torch.cat([cls_token_proj.expand(-1, pcg_tokens_proj.size(1), -1), pcg_tokens_proj], dim=-1)  # (batch_size, L, token_dim * 2)
        
        # Multiscale Convolution
        # conv를 위해 차원 변경
        ecg_concat = ecg_concat.transpose(1, 2)  # (batch_size, token_dim * 2, L)
        pcg_concat = pcg_concat.transpose(1, 2)  # (batch_size, token_dim * 2, L)

        # 다중 크기의 컨볼루션 적용
        ecg_out3 = self.conv3(ecg_concat)
        ecg_out5 = self.conv5(ecg_concat)
        ecg_out7 = self.conv7(ecg_concat)
        
        pcg_out3 = self.conv3(pcg_concat)
        pcg_out5 = self.conv5(pcg_concat)
        pcg_out7 = self.conv7(pcg_concat)

        # 세 개의 컨볼루션 결과를 Concat
        ecg_multiscale = torch.cat([ecg_out3, ecg_out5, ecg_out7], dim=1)  # (batch_size, token_dim * 3, L)
        pcg_multiscale = torch.cat([pcg_out3, pcg_out5, pcg_out7], dim=1)  # (batch_size, token_dim * 3, L)
        
        ecg_multiscale = self.conv1x1(ecg_multiscale).transpose(1, 2)  # (batch_size, L, token_dim)
        pcg_multiscale = self.conv1x1(pcg_multiscale).transpose(1, 2)  # (batch_size, L, token_dim)

        # LayerNorm
        ecg_multiscale = self.norm1(ecg_multiscale)
        pcg_multiscale = self.norm1(pcg_multiscale)

        # GELU 
        ecg_features = self.gelu(ecg_multiscale)
        pcg_features = self.gelu(pcg_multiscale)

        # Residual Connection
        # 여기서 부터 작성
        
        return ecg_output, pcg_output






