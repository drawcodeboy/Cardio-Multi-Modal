import torch

def generate_cls_token(ecg_tokens, pcg_tokens):
    # 1. ECG, PCG -> 각각 GAP 적용
    ecg_cls = torch.mean(ecg_tokens, dim=-1, keepdim=True)  # (batch_size, token_dim, 1)
    pcg_cls = torch.mean(pcg_tokens, dim=-1, keepdim=True)  # (batch_size, token_dim, 1)

    # 2. 차원에 맞게 concat
    concat_features = torch.cat([ecg_cls, pcg_cls], dim=-1)  # (batch_size, token_dim, 2)
    
    # 3. 다시 한번 GAP 적용 -> LX1 vector의 Cls token 생성
    cls_token = torch.mean(concat_features, dim=-1, keepdim=True)  # (batch_size, token_dim, 1)

    return cls_token