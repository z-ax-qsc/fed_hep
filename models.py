from torch import nn
import pennylane as qml
import torch
from params import IsQuantum

# Define your QLSTM and model classes (same as before)
class QLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_qubits=4,
                 n_qlayers=1,
                 batch_first=True,
                 backend="default.qubit"):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend

        # Initialize the quantum device
        self.dev = qml.device(self.backend, wires=self.n_qubits)

        # Define wires
        self.wires = list(range(self.n_qubits))

        # Define weight shapes
        self.weight_shapes = {"weights": (self.n_qlayers, self.n_qubits, 3)}

        # Define classical layers
        self.clayer_in = nn.Linear(self.n_inputs + self.hidden_size, self.n_qubits)
        self.clayer_out = nn.Linear(self.n_qubits, self.hidden_size)

        # Define the quantum node (QNode)
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")

        # Quantum circuit layers
        self.VQC = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)

        self.LSTM = nn.Sequential(nn.Linear(self.n_qubits, self.n_qubits))
        for l in range(1,self.n_qlayers):
          self.LSTM.append(nn.Linear(self.n_qubits, self.n_qubits))

    def circuit(self, inputs, weights):
        # Encoding layer
        qml.templates.AngleEmbedding(inputs, wires=self.wires)
        # Variational layers
        qml.templates.StronglyEntanglingLayers(weights, wires=self.wires)
        # Measurement
        return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires]

    def forward(self, x, init_states=None):
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size)

        hidden_seq = []

        for t in range(seq_length):
            x_t = x[:, t, :]
            v_t = torch.cat((h_t, x_t), dim=1)
            y_t = self.clayer_in(v_t)
            if IsQuantum:
              y_t = self.VQC(y_t)
            else:
              y_t = self.LSTM(y_t)
            y_t = self.clayer_out(y_t)

            # LSTM gates
            f_t = torch.sigmoid(y_t)
            i_t = torch.sigmoid(y_t)
            g_t = torch.tanh(y_t)
            o_t = torch.sigmoid(y_t)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)

class QShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, n_qubits=4, n_qlayers=1, backend='default.qubit'):
        super().__init__()
        self.num_sensors = num_sensors  # Number of features
        self.hidden_units = hidden_units

        self.lstm = QLSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
            backend=backend
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.linear(hn).flatten()
        return out
