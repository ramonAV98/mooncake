import torch
from torch import nn
from .base import BaseTrainer

AF = {"relu": nn.ReLU, "identity": lambda x: x}


class FeedForward(nn.Module):
    def __init__(self, n_layers=20, activation_functions="relu", n_features=1,
                 neurons_per_layer=20, last_activiation_function="identity"):
        super().__init__()
        layers = self.create_layers(n_layers, activation_functions, n_features,
                                    neurons_per_layer,
                                    last_activiation_function)
        self.sequential = nn.Sequential(layers)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, x):
        return self.sequential[x]

    @staticmethod
    def create_layers(n_layers, activation_functions, n_features,
                      neurons_per_layer,
                      last_activiation_function):
        if type(activation_functions) == str:
            activation_functions = [activation_functions]

        # map activation functions
        new_af = []
        for activation_function in activation_functions:
            new_af.append(AF[activation_function])
        new_af.append(AF[last_activiation_function])
        activation_functions = new_af

        # create layers
        layers = []
        layers.append(nn.Linear(n_features, neurons_per_layer))
        layers.append(activation_functions[0]())
        for i in range(1, n_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(activation_functions[i]())
        return layers


class FeedForwardTrainer(BaseTrainer):
    def __init__(self, max_epochs=15, batch_size=32, lr=1e-4,
                 max_lr=1e-3, input_seq_len=5, output_seq_len=5, step=1,
                 module__emb_dim=32, module__emb_sizes=(1,),
                 module__enc_size=10, module__dec_size=10,
                 module__hidden_size=48, module__tf_ratio=0.2,
                 grad_clipping=True,
                 solver='adam', scheduler='one_cycle_lr', **others):
        super().__init__(solver=solver, scheduler=scheduler,
                         max_epochs=max_epochs, batch_size=batch_size, lr=lr,
                         grad_clipping=grad_clipping, max_lr=max_lr,
                         module__emb_sizes=module__emb_sizes,
                         module__emb_dim=module__emb_dim,
                         module__enc_size=module__enc_size,
                         module__dec_size=module__dec_size,
                         module__hidden_size=module__hidden_size,
                         module__tf_ratio=module__tf_ratio,
                         module__cell_type=module__cell_type)
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.step = step