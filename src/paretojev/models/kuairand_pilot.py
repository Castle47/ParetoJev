"""Memory-bounded SASRec-style + MMoE for seven observed feedback targets."""
from __future__ import annotations
import torch
from torch import nn
from .next_item import NextItemConfig, SASRecNextItem


class KuaiRandPilotModel(nn.Module):
    def __init__(self, config, *, profile_cardinalities=()):
        super().__init__()
        h = config['hidden_dim']
        self.sequence = SASRecNextItem(NextItemConfig(num_items=config['item_hash_buckets'] + 2,
            max_length=config['history_length'], hidden=h, heads=config['heads'],
            layers=config['layers'], dropout=0.1))
        self.users = nn.Embedding(27286, 8, padding_idx=0)
        self.register_buffer('known_users', torch.zeros(27285, dtype=torch.bool))
        self.tabs = nn.Embedding(15, 4)
        self.profile = nn.ModuleList(nn.Embedding(n, 4, padding_idx=0) for n in profile_cardinalities)
        d = h * 3 + 8 + 4 + 7 + 4 * len(profile_cardinalities)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU()) for _ in range(config['experts']))
        self.gates = nn.ModuleList(nn.Linear(d, config['experts']) for _ in range(7))
        self.heads = nn.ModuleList(nn.Linear(32, 1) for _ in range(7))

    def set_training_users(self, users):
        self.known_users.zero_()
        self.known_users[torch.as_tensor(users, dtype=torch.long, device=self.known_users.device)] = True

    def forward(self, history, item, user, tab, numeric, profile=None):
        states = self.sequence.states(history)
        last = (history.ne(0).sum(dim=1) - 1).clamp_min(0)
        state = states[torch.arange(len(history), device=history.device), last]
        candidate = self.sequence.items(item)
        fields = [state, candidate, state * candidate, self.users(torch.where(self.known_users[user], user + 1, 0)), self.tabs(tab), numeric]
        if self.profile:
            if profile is None or profile.shape[1] != len(self.profile):
                raise ValueError('Profile fields required for registered sensitivity regime')
            fields.extend(layer(profile[:, i]) for i, layer in enumerate(self.profile))
        elif profile is not None:
            raise ValueError('Profiles forbidden in event-only regime')
        x = torch.cat(fields, dim=1)
        experts = torch.stack([layer(x) for layer in self.experts], dim=1)
        return torch.cat([head((torch.softmax(gate(x), dim=-1)[..., None] * experts).sum(dim=1))
                          for head, gate in zip(self.heads, self.gates)], dim=1)
