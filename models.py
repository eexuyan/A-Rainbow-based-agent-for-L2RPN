import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Inspired from https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb
class NoisyLayer(nn.Module):
    def __init__(self, in_dim, out_dim, params=None, is_noisy=False):
        super(NoisyLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = params
        self.is_noisy = is_noisy
        self.sigma_init = 0.5
        self.mu_w = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.mu_b = nn.Parameter(torch.Tensor(out_dim))
        self.sigma_w = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.sigma_b = nn.Parameter(torch.Tensor(out_dim))
        # Epsilon is not trainable
        self.register_buffer("eps_w", torch.Tensor(out_dim, in_dim))
        self.register_buffer("eps_b", torch.Tensor(out_dim))
        self.init_params()
        self.update_noise()

    def init_params(self):
        # Trainable params
        nn.init.uniform_(self.mu_w, -math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim))
        nn.init.uniform_(self.mu_b, -math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim))
        nn.init.constant_(self.sigma_w, self.sigma_init / math.sqrt(self.out_dim))
        nn.init.constant_(self.sigma_b, self.sigma_init / math.sqrt(self.out_dim))

    def update_noise(self):
        self.eps_w.copy_(self.factorize_noise(self.out_dim).ger(self.factorize_noise(self.in_dim)))
        self.eps_b.copy_(self.factorize_noise(self.out_dim))

    def factorize_noise(self, size):
        # Modify scale to amplify or reduce noise
        x = torch.Tensor(np.random.normal(loc=0.0, scale=0.001, size=size))
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        return F.linear(x, self.mu_w + self.sigma_w * self.eps_w, self.mu_b + self.sigma_b * self.eps_b)


class DuelingNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, params=None, is_noisy=False):
        super(DuelingNetwork, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim, 256),
                                         nn.ReLU())
        self.params = params

        if is_noisy:
            self.advantage_layer_hidden = NoisyLayer(256, 256)
            self.advantage_layer_out = NoisyLayer(256, out_dim)

            self.value_layer_hidden = NoisyLayer(256, 256)
            self.value_layer_out = NoisyLayer(256, 1)
        else:
            self.advantage_layer_hidden = nn.Linear(256, 256)
            self.advantage_layer_out = nn.Linear(256, out_dim)

            self.value_layer_hidden = nn.Linear(256, 256)
            self.value_layer_out = nn.Linear(256, 1)

        self.advantag_act = nn.ReLU()
        self.value_layer_act = nn.ReLU()


    def update_noise(self):
        self.advantage_layer_hidden.update_noise()
        self.advantage_layer_out.update_noise()

        self.value_layer_hidden.update_noise()
        self.value_layer_out.update_noise()

    def forward(self, x):
        x = self.input_layer(x)

        # Advantage
        advantage = self.advantage_layer_hidden(x)
        advantage = self.advantag_act(advantage)
        advantage = self.advantage_layer_out(advantage)

        # Value
        value = self.value_layer_hidden(x)
        value = self.value_layer_act(value)
        value = self.value_layer_out(value)

        return self.softmax(advantage + value - advantage.mean())


class DistributionalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, distr_params, is_noisy):
        super(DistributionalNetwork, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_bins = distr_params["num_bins"]
        self.v_range = distr_params["v_range"]
        if is_noisy:
            self.layers = nn.Sequential(nn.Linear(in_dim, 256),
                                        nn.ReLU())
            self.hidden_layer = NoisyLayer(256, 256)
            self.hidden_act = nn.ReLU()
            self.output = NoisyLayer(256, self.out_dim * self.num_bins)
        else:
            self.layers = nn.Sequential(nn.Linear(in_dim, 256),
                                        nn.ReLU())
            self.hidden_layer = nn.Linear(256, 256)
            self.hidden_act = nn.ReLU()
            self.output = nn.Linear(256, self.out_dim * self.num_bins)


    def action_distr(self, x):
        x = self.layers(x)
        x = self.hidden_layer(x)
        x = self.output(x)
        x = x.reshape(-1, self.out_dim, self.num_bins)
        return nn.Softmax(dim=2)(x).clamp(1e-5)

    def update_noise(self):
        self.hidden_layer.update_noise()
        self.output.update_noise()

    def forward(self, x, possible_moves):
        x = torch.sum(self.action_distr(x) * self.v_range, dim=2)
        return x


class DuelingDistributionalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, distr_params, is_noisy):
        super(DuelingDistributionalNetwork, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_dim, 400),
                                         nn.ReLU())
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_bins = distr_params["num_bins"]
        self.v_range = distr_params["v_range"]
        
        if is_noisy:
            self.advantage_layer_hidden = NoisyLayer(400, 400)
            self.advantage_layer_out = NoisyLayer(400, out_dim * self.num_bins)

            self.value_layer_hidden = NoisyLayer(400, 400)
            self.value_layer_out = NoisyLayer(400, self.num_bins)
        else:
            self.advantage_layer_hidden = nn.Linear(400, 400)
            self.advantage_layer_out = nn.Linear(400, out_dim * self.num_bins)

            self.value_layer_hidden = nn.Linear(400, 400)
            self.value_layer_out = nn.Linear(400, self.num_bins)

        self.advantage_act = nn.ReLU()
        self.value_layer_act = nn.ReLU()

        self.softmax = nn.Softmax(dim=2)
        
    def update_noise(self):
        self.advantage_layer_hidden.update_noise()
        self.advantage_layer_out.update_noise()

        self.value_layer_hidden.update_noise()
        self.value_layer_out.update_noise()

    def action_distr(self, x):
        x = self.input_layer(x)

        # Advantage
        advantage = self.advantage_layer_hidden(x)
        advantage = self.advantage_act(advantage)

        # Value
        value = self.value_layer_hidden(x)
        value = self.value_layer_act(value)

        advantage = self.advantage_layer_out(advantage).reshape(-1, self.out_dim, self.num_bins)
        value = self.value_layer_out(value).reshape(-1, 1, self.num_bins)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.softmax(q).clamp(1e-5)

    def forward(self, x):
        x = torch.sum(self.action_distr(x) * self.v_range, dim=2)
        return x
    
    