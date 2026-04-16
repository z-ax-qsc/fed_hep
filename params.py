import torch
from torch import nn

# Map optimizer and loss function names to actual classes
optimizer_dict = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
}

loss_function_dict = {
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
}

# Define hyperparameters
hyperparameter_space = {
    'learning_rate': [], 
    'batch_size': [],
    'sequence_length': [],
    'num_hidden_units': [],
    'num_qubits': [],
    'num_qlayers': [],
    'optimizer': [], # one from optimizer_dict 
    'loss_function': [], # One of loss_function_dict
    'backend': [], # Pennylane backend to use eg. 'default.qubit'
    'num_epochs': [],  # Local epochs
    'num_nodes': [],
    'sample_size': [], # Samples from data
    'isSelectedColumns': [], # True or False
}

# Number of global communication rounds
num_rounds =  ; # Adjust as needed

IsQuantum = ; # True or False