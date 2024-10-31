import torch
import torch.nn as nn
import torch.nn.functional as F

class MCT_VIT_Block(nn.Module):
    def __init__(self, token_dim = 32, num_heads=8): # 멀티헤드 8개로 나눔
        super(MCT_VIT_Block, self).__init__()
        # 1. Linear(차원을 맞춰주기 위해)
        self.cls_linear = nn.Linear(token_dim , token_dim//2)
        self.token_linear = nn.Linear(token_dim, token_dim//2)
        
        # 2. Multiscale Conv
        self.conv3 = nn.Conv1d(token_dim, token_dim, kernel_size=3, padding=1) # 동일한 출력을 위해 padding 조절
        self.conv5 = nn.Conv1d(token_dim, token_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(token_dim, token_dim, kernel_size=7, padding=3)

        # 3. Feed-forward(1)
        self.norm_conv = nn.LayerNorm(token_dim)
        self.conv1x1 = nn.Conv1d(token_dim, token_dim, kernel_size=1)
        self.norm_ffn = nn.LayerNorm(token_dim)
        self.gelu = nn.GELU()

        # 4. Feed-forward(2)-(MHSA)
        self.norm_mhsa = nn.LayerNorm(token_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim = token_dim, num_heads = 8, batch_first = True)

    def forward(self, input_tokens, cls_token):
        '''
        input
        cls_token_shape: (128, 32, 1) = (batch_size, token_dim, sequence_length)
        ecg_tokens_shape: torch.Size([128, 32, 313]), pcg_tokens_shape: torch.Size([128, 32, 313]) = (batch_size, token_dim, sequence_length)
        '''
        # 1. Linear
        cls_token = cls_token.permute(0, 2, 1)  # (batch_size, 1, token_dim) 
        input_tokens = input_tokens.permute(0, 2, 1)  # (batch_size, sequence_length, token_dim) 

        cls_token_proj = self.cls_linear(cls_token)  # (batch_size, 1, token_dim/2)
        input_tokens_proj = self.token_linear(input_tokens)  # (batch_size, sequence_length, token_dim/2)

        # concat(Cls token 벡터를 시퀀스의 길이만큼 repeat시킴)
        token_concat = torch.cat([cls_token_proj.expand(-1, input_tokens_proj.size(1), -1), input_tokens_proj], dim=-1)  # (batch_size, sequence_length, token_dim)
        
        # 2. Multiscale Convolution
        token_concat = token_concat.transpose(1, 2)  # (batch_size, token_dim, sequence_length)  

        token_out3 = self.conv3(token_concat)   # (batch_size, token_dim, sequence_length)  
        token_out5 = self.conv5(token_concat)
        token_out7 = self.conv7(token_concat)

        token_multiscale = token_concat + token_out3 + token_out5 + token_out7 # (batch_size, token_dim, sequence_length) 

        # 3. Feed-forward(1)
        token_multiscale = token_multiscale.permute(0, 2, 1)  
        norm_multiscale = self.norm_conv(token_multiscale) # (batch_size, sequence_length, token_dim) 

        conv_input = norm_multiscale.permute(0, 2, 1)  
        conv_output = self.conv1x1(conv_input)  # (batch_size, token_dim, sequence_length)
        conv_output = conv_output.permute(0, 2, 1) 

        norm_ffn = self.norm_ffn(conv_output) # (batch_size, sequence_length, token_dim) 
        activated_ffn = self.gelu(norm_ffn)  # (batch_size, sequence_length, token_dim)
        token_residual = token_multiscale + activated_ffn # (batch_size, sequence_length, token_dim)

        # 4. Feed-forward(2)-(MHSA)
        norm_mhsa_input = self.norm_mhsa(token_residual) # (batch_size, sequence_length, token_dim)
        attn_output, _ = self.mhsa(norm_mhsa_input, norm_mhsa_input, norm_mhsa_input)  # (batch_size, sequence_length, token_dim)

        final_output = attn_output.permute(0, 2, 1)  # (batch_size, 1, token_dim) 

        # output: (batch_size, token_dim, sequence_length) 
        return final_output






