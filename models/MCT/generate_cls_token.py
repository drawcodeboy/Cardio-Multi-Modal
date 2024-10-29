import torch

def generate_cls_token(ecg_tokens, pcg_tokens):
    """
    1. ECG, PCG -> 각각 GAP 적용
    2. 차원에 맞게 concat
    3. 다시 한번 GAP 적용 -> LX1 vector의 Cls token 생성
    """

    ecg_cls = torch.mean(ecg_tokens, dim=-1, keepdim=True)  # (batch_size, L, 1)
    pcg_cls = torch.mean(pcg_tokens, dim=-1, keepdim=True)  # (batch_size, L, 1)

    print(f'ecg_cls_shape = {ecg_cls.shape}, pcg_shape = {pcg_cls.shape}')

    concat_features = torch.cat([ecg_cls, pcg_cls], dim=-1)  # (batch_size, L, 2)

    print(f'concat_features_shape = {concat_features.shape}')

    
    cls_token = torch.mean(concat_features, dim=-1, keepdim=True)  # (batch_size, L, 1)
    print(f'cls_token_shape = {cls_token.shape}')

    
    return cls_token

