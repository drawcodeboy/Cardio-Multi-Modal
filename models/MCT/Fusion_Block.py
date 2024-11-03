import torch
from torch import nn

class FusionBlock(nn.Module):
    def __init__(self, token_dim=32, lambda_param=0.5):
        super(FusionBlock, self).__init__()
        self.Wq = nn.Linear(token_dim, token_dim//2)  # Query 
        self.Wk = nn.Linear(token_dim, token_dim//2)  # Key 

    def forward(self, cls_token, ecg_features, pcg_features):
        '''
        # input
        cls_token_shape: (batch_size, 32, 1) (batch_size, token_dim, 1)
        ecg_feature_output_shape: (batch_size, token_dim, sequence_length) (128, 32, 313)
        '''
        # cls_token을 Query 벡터(Qi)로 변환
        Q = self.Wq(cls_token.transpose(1, 2)).transpose(1, 2)  # (batch_size, token_dim/2, 1)

        # ECG와 PCG 특징을 Concatenate하여 Key 벡터(Ki+1)로 변환
        combined_features = torch.cat((ecg_features, pcg_features), dim=2)  # (batch_size, token_dim, 2 * sequence_length)
        K = self.Wk(combined_features.transpose(1, 2)).transpose(1, 2)  # (batch_size, token_dim / 2, 2 * sequence_length)

        # Q와 K의 내적을 계산하여 통합된 중요도 S를 구함
        S = torch.bmm(Q.transpose(1, 2), K)  # (batch_size, 1, 2 * sequence_length)

        # S 정렬하기
        S = S.view(S.size(0), 2, -1)  # (batch_size, 2, sequence_length)

        # X_prime_i_plus 구하기
        S_e, S_p = S[:, 0, :], S[:, 1, :]  # (batch_size, sequence_length)로 각각 분할

        S_e = S_e.unsqueeze(1)  # (batch_size, 1, sequence_length) 각각 분할
        S_p = S_p.unsqueeze(1)  # (batch_size, 1, sequence_length)

        weighted_ecg_features = S_e * ecg_features  # (batch_size, dim, length)
        weighted_pcg_features = S_p * pcg_features  # (batch_size, dim, length)

        X_prime_i_plus_1 = weighted_ecg_features + weighted_pcg_features  # (batch_size, dim, length)

        # K_prime_i_puls_1 구하기
        K_prime_i_puls_1 = self.Wk(X_prime_i_plus_1.transpose(1, 2)).transpose(1, 2)

        # S_prime_i_puls_1 구하기
        S_prime_i_puls_1 = torch.bmm(Q.transpose(1, 2), K_prime_i_puls_1)  # (batch_size, 1, length)

        # cls_prime_i_puls_1 구하기
        S_prime_i_plus_1_expanded = S_prime_i_puls_1.expand_as(X_prime_i_plus_1)  # (batch_size, dim, length)

        cls_prime_i_plus_1 = torch.sum(S_prime_i_plus_1_expanded * X_prime_i_plus_1, dim=-1).unsqueeze(-1)  # (batch_size, dim, 1)

        output = 0.5 * cls_token + 0.5 * cls_prime_i_plus_1  # (batch_size, dim, 1)

        return output
