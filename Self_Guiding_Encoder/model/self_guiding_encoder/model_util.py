# implemented by PowerHu
import torch
import numpy as np
import math
import scipy.sparse as sp


##############################################
# beta schedule functions
##############################################
def get_schedule(num_steps, schedule):
    if schedule == 'quad':
        return quad_schedule(num_steps)
    elif schedule == 'linear':
        return linear_schedule(num_steps)
    elif schedule == 'cosine':
        return cosine_schedule(num_steps)
    else:
        raise ValueError(f'Unknown schedule {schedule}')

def quad_schedule(num_steps, beta_start=0.0001, beta_end=0.5):
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, dtype=torch.float32) ** 2
    return betas

def linear_schedule(num_steps, beta_start=0.0001, beta_end=0.5):
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    return betas

def cosine_schedule(timesteps, s=0.008):
    '''
    cosine schedule from https://openreview.net/forum?id=-NEXDKk8gZ
    '''
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

#############################
# positional embeddings
#############################
def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def laplacian_positional_encoding(adj, pos_enc_dim):
    L = calculate_normalized_laplacian(adj)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = np.abs(np.real(EigVal)).argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    if pos_enc_dim > 0:
        if not EigVec.shape[1] >= pos_enc_dim:
            padding = np.zeros((EigVec.shape[0], pos_enc_dim - EigVec.shape[1]))
            pos_enc = np.concatenate((EigVec, padding), axis=1)
        else:
            pos_enc = EigVec[:, :pos_enc_dim]
    else:
        pos_enc = EigVec
    return pos_enc

def temporal_positional_embedding(t_len, d_model=128):
    tpe = np.zeros([t_len, d_model])
    position = np.arange(0, t_len)[:, np.newaxis]
    div_term = 1 / np.power(
        10000.0, np.arange(0, d_model, 2) / d_model
    )
    tpe[:, 0::2] = np.sin(position * div_term)
    tpe[:, 1::2] = np.cos(position * div_term)
    return tpe


def norm_adj(adj):
    return [(adj / adj.sum(dim=-1, keepdim=True)),
            (adj.t() / adj.t().sum(dim=-1, keepdim=True))]