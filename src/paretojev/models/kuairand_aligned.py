"""Common-information MMoE and its explicitly richer sequence/ID extension."""
import numpy as np
import torch
from torch import nn
from .next_item import NextItemConfig, SASRecNextItem


def fit_normalization(x, weights):
    """Train-only weighted moments; missing values become zero after scaling."""
    w = np.asarray(weights, dtype='float64')
    means, scales = [], []
    for col in x.T:
        valid = np.isfinite(col)
        if not valid.any():
            means.append(0.); scales.append(1.); continue
        v = col[valid].astype('float64'); ww = w[valid]
        mean = np.average(v, weights=ww)
        std = np.sqrt(np.average((v - mean)**2, weights=ww))
        means.append(mean); scales.append(max(std, 1e-6))
    return np.array(means, dtype='float32'), np.array(scales, dtype='float32')


class AlignedBehaviorModel(nn.Module):
    def __init__(self, config, regime):
        super().__init__()
        if regime not in ('tabular_mmoe', 'sasrec_mmoe'):
            raise ValueError('Unknown regime')
        self.regime = regime
        width = config['columns'] - 1
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('scale', torch.ones(width))
        self.tabs = nn.Embedding(15, 4)
        d = 2 * width + 4
        if regime == 'sasrec_mmoe':
            h = config['hidden_dim']
            self.sequence = SASRecNextItem(NextItemConfig(
                num_items=config['item_hash_buckets']+2, max_length=config['history_length'],
                hidden=h, heads=config['heads'], layers=config['layers'], dropout=.1))
            self.users = nn.Embedding(27286, 8, padding_idx=0)
            self.register_buffer('known_users', torch.zeros(27285, dtype=torch.bool))
            d += 3*h+8
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU()) for _ in range(config['experts']))
        self.gates = nn.ModuleList(nn.Linear(d, config['experts']) for _ in range(7))
        self.heads = nn.ModuleList(nn.Linear(32, 1) for _ in range(7))

    def forward(self, x, history=None, item=None, user=None):
        numeric = x[:, 1:]
        valid = torch.isfinite(numeric)
        scaled = torch.where(valid, (numeric-self.mean)/self.scale, 0.)
        fields = [self.tabs(x[:, 0].long()), scaled, (~valid).float()]
        if self.regime == 'sasrec_mmoe':
            if history is None or item is None or user is None:
                raise ValueError('Sequence regime requires history, candidate and user')
            states = self.sequence.states(history)
            last = (history.ne(0).sum(1)-1).clamp_min(0)
            state = states[torch.arange(len(history), device=x.device), last]
            candidate = self.sequence.items(item)
            fields += [state, candidate, state*candidate,
                self.users(torch.where(self.known_users[user], user+1, 0))]
        elif any(v is not None for v in (history, item, user)):
            raise ValueError('IDs forbidden in identical-information tabular arm')
        features = torch.cat(fields, 1)
        experts = torch.stack([layer(features) for layer in self.experts], 1)
        return torch.cat([head((gate(features).softmax(-1)[..., None]*experts).sum(1))
            for gate, head in zip(self.gates, self.heads)], 1)

    @torch.no_grad()
    def reset_reserved(self):
        if self.regime == 'sasrec_mmoe':
            self.sequence.reset_reserved()
            self.users.weight[0].zero_()
