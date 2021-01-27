"""Example models for training protein sequence to angle coordinates tasks."""

import numpy as np
import torch
from sidechainnet.structure.build_info import NUM_ANGLES
import torch.nn.functional as F
import torch.nn as nn
import bcolz


class BaseProteinAngleRNN(torch.nn.Module):
    """A simple RNN that consumes protein sequences and produces angles."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=20,
                 n_angles=NUM_ANGLES,
                 bidirectional=True,
                 sincos_output=True,
                 device=torch.device('cpu')):
        super(BaseProteinAngleRNN, self).__init__()
        self.size = size
        self.n_layers = n_layers
        self.sincos_output = sincos_output
        self.d_out = n_angles * 2 if sincos_output else n_angles
        self.lstm = torch.nn.LSTM(d_in,
                                  size,
                                  n_layers,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        self.n_direction = 2 if bidirectional else 1
        self.hidden2out = torch.nn.Linear(self.n_direction * size, self.d_out)
        self.output_activation = torch.nn.Tanh()
        self.device_ = device

    def init_hidden(self, batch_size):
        """Initialize the hidden state vectors at the start of a batch iteration."""
        h, c = (torch.zeros(self.n_layers * self.n_direction, batch_size,
                            self.size).to(self.device_),
                torch.zeros(self.n_layers * self.n_direction, batch_size,
                            self.size).to(self.device_))
        return h, c

    def forward(self, *args, **kwargs):
        """Run one forward step of the model."""
        raise NotImplementedError


class IntegerSequenceProteinRNN(BaseProteinAngleRNN):
    """A protein sequence-to-angle model that consumes integer-coded sequences."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=20,
                 n_angles=NUM_ANGLES,
                 bidirectional=True,
                 device=torch.device('cpu'),
                 sincos_output=True):
        super(IntegerSequenceProteinRNN, self).__init__(size=size,
                                                        n_layers=n_layers,
                                                        d_in=d_in,
                                                        n_angles=n_angles,
                                                        bidirectional=bidirectional,
                                                        device=device,
                                                        sincos_output=sincos_output)

        self.input_embedding = torch.nn.Embedding(21, 20, padding_idx=20)

    def forward(self, sequence):
        """Run one forward step of the model."""
        # First, we compute the length of each sequence to use pack_padded_sequence
        lengths = sequence.shape[-1] - (sequence == 20).sum(axis=1)
        h, c = self.init_hidden(len(lengths))
        # Our inputs are sequences of integers, allowing us to use torch.nn.Embeddings
        sequence = self.input_embedding(sequence)
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                           lengths.cpu(),
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                                        batch_first=True)
        output = self.hidden2out(output)
        if self.sincos_output:
            output = self.output_activation(output)
            output = output.view(output.shape[0], output.shape[1], int(self.d_out / 2), 2)
        else:
            # We push the output through a tanh layer and multiply by pi to ensure
            # values are within [-pi, pi] for predicting raw angles.
            output = self.output_activation(output) * np.pi
            output = output.view(output.shape[0], output.shape[1], self.d_out)
        return output


class PSSMProteinRNN(BaseProteinAngleRNN):
    """A protein structure model consuming 1-hot sequences, 2-ary structures, & PSSMs."""

    def __init__(self,
                 size,
                 n_layers=1,
                 d_in=49,
                 n_angles=NUM_ANGLES,
                 bidirectional=True,
                 device=torch.device('cpu'),
                 sincos_output=True):
        """Create a PSSMProteinRNN model with input dimensionality 41."""
        super(PSSMProteinRNN, self).__init__(size=size,
                                             n_layers=n_layers,
                                             d_in=d_in,
                                             n_angles=n_angles,
                                             bidirectional=bidirectional,
                                             device=device,
                                             sincos_output=sincos_output)

    def forward(self, sequence):
        """Run one forward step of the model."""
        # First, we compute the length of each sequence to use pack_padded_sequence
        lengths = sequence.shape[1] - (sequence == 0).all(axis=2).sum(axis=1)
        h, c = self.init_hidden(len(lengths))
        sequence = torch.nn.utils.rnn.pack_padded_sequence(sequence,
                                                           lengths.cpu(),
                                                           batch_first=True,
                                                           enforce_sorted=False)
        output, (h, c) = self.lstm(sequence, (h, c))
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output,
                                                                        batch_first=True)
        output = self.hidden2out(output)
        if self.sincos_output:
            output = self.output_activation(output)
            output = output.view(output.shape[0], output.shape[1], int(self.d_out / 2), 2)
        else:
            # We push the output through a tanh layer and multiply by pi to ensure
            # values are within [-pi, pi] for predicting raw angles.
            output = self.output_activation(output) * np.pi
            output = output.view(output.shape[0], output.shape[1], self.d_out)
        return output
class RGN(nn.Module):
    def __init__(self, hidden_size, num_layers, linear_units=20, input_size=42):
        super(RGN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.linear_units = linear_units
        self.grads = {}
        
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, bidirectional=True)
        
        #initialize alphabet to random values between -pi and pi
        u = torch.distributions.Uniform(-3.14, 3.14)
        self.alphabet = nn.Parameter(u.rsample(torch.Size([linear_units, 3])))
        self.linear = nn.Linear(hidden_size*2, linear_units)
        
        #set coordinates for first 3 atoms to identity matrix
        self.A = torch.tensor([0., 0., 1.])
        self.B = torch.tensor([0., 1., 0.])
        self.C = torch.tensor([1., 0., 0.])

        #bond length vectors C-N, N-CA, CA-C
        self.avg_bond_lens = torch.tensor([1.329, 1.459, 1.525])
        #bond angle vector, in radians, CA-C-N, C-N-CA, N-CA-C
        self.avg_bond_angles = torch.tensor([2.034, 2.119, 1.937])

    
    def forward(self, sequences, lengths):
        max_len = sequences.size(0)
        batch_sz = sequences.size(1)
        lengths = torch.tensor(lengths, dtype=torch.long, requires_grad=False)
        order = [x for x,y in sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)]
        conv = zip(range(batch_sz), order) #for unorder after LSTM
        
        #add absolute position of residue to the input vector
        abs_pos = torch.tensor(range(max_len), dtype=torch.float32).unsqueeze(1)
        abs_pos = (abs_pos * torch.ones((1, batch_sz))).unsqueeze(2) #broadcasting
        
        h0 = Variable(torch.zeros((self.num_layers*2, batch_sz, self.hidden_size)))
        c0 = Variable(torch.zeros((self.num_layers*2, batch_sz, self.hidden_size)))
        
        #input needs to be float32 and require grad
        sequences = torch.tensor(sequences, dtype=torch.float32, requires_grad=True)
        pad_seq = torch.cat([sequences, abs_pos], 2)
    
        packed = pack_padded_sequence(pad_seq[:, order], lengths[order], batch_first=False)
        
        lstm_out, _ = self.lstm(packed, (h0,c0))
        unpacked, _ = pad_packed_sequence(lstm_out, batch_first=False, padding_value=0.0)
        
        #reorder back to original to match target
        reorder = [x for x,y in sorted(conv, key=lambda x: x[1], reverse=False)]
        unpacked = unpacked[:, reorder]

        #for example, see https://bit.ly/2lXJC4m
        softmax_out = F.softmax(self.linear(unpacked), dim=2)
        sine = torch.matmul(softmax_out, torch.sin(self.alphabet))
        cosine = torch.matmul(softmax_out, torch.cos(self.alphabet))
        out = torch.atan2(sine, cosine)
        
        #create as many copies of first 3 coords as there are samples in the batch
        broadcast = torch.ones((batch_sz, 3))
        pred_coords = torch.stack([self.A*broadcast, self.B*broadcast, self.C*broadcast])
        
        for ix, triplet in enumerate(out[1:]):
            pred_coords = geometric_unit(pred_coords, triplet, 
                                         self.avg_bond_angles, 
                                         self.avg_bond_lens)
        #pred_coords.register_hook(self.save_grad('pc'))
        
            
        #pdb.set_trace()
        return pred_coords
    
    def save_grad(self, name):
        def hook(grad): self.grads[name] = grad
        return hook    


def geometric_unit(pred_coords, pred_torsions, bond_angles, bond_lens):
    for i in range(3):
        #coordinates of last three atoms
        A, B, C = pred_coords[-3], pred_coords[-2], pred_coords[-1]
        
        #internal coordinates
        T = bond_angles[i]
        R = bond_lens[i]
        P = pred_torsions[:, i]

        #6x3 one triplet for each sample in the batch
        D2 = torch.stack([-R*torch.ones(P.size())*torch.cos(T), 
                          R*torch.cos(P)*torch.sin(T),
                          R*torch.sin(P)*torch.sin(T)], dim=1)

        #bsx3 one triplet for each sample in the batch
        BC = C - B
        bc = BC/torch.norm(BC, 2, dim=1, keepdim=True)

        AB = B - A

        N = torch.cross(AB, bc)
        n = N/torch.norm(N, 2, dim=1, keepdim=True)

        M = torch.stack([bc, torch.cross(n, bc), n], dim=2)

        D = torch.bmm(M, D2.view(-1,3,1)).squeeze() + C
        pred_coords = torch.cat([pred_coords, D.view(1,-1,3)])
    
    return pred_coords

def pair_dist(x, y=None):
    #norm takes sqrt, undo that by squaring
    #x_norm = torch.pow(torch.norm(x, 2), 2).view(-1,1)
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.t(y)
        #y_norm = torch.pow(torch.norm(x, 2), 2).view(1,-1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.t(x)
        y_norm = x_norm.view(1,-1)
    
    dist = x_norm + y_norm - 2*torch.mm(x, y_t)
    if y is None:
        #enforce all zeros along the diagonal
        dist = dist - torch.diag(torch.diag(dist))
    dist = F.relu(dist)
    
    return torch.pow(dist, 0.5)

def batch_pair_dist(x):
    x = x.permute(dims=(1,0,2))
    x_norm = (x**2).sum(2).view(x.size(0), -1, 1)
    
    y_t = x.permute(0,2,1)
    y_norm = x_norm.view(x.size(0), 1, -1)
    
    dist = x_norm + y_norm - 2*torch.bmm(x, y_t)
    dist = F.relu(dist)
    
    return torch.pow(dist, 0.5)

class dRMSD(nn.Module):
    def __init__(self):
        super(dRMSD, self).__init__()

    def forward(self, x, y, mask):
        #put batch on the first dimension
        x = x.permute(dims=(1,0,2))
        y = y.permute(dims=(1,0,2))
        mask = mask.permute(dims=(1,0))
        
        dRMSD = torch.tensor([0.])
        for i in range(x.size(0)):
            #3 to exclude random first 3 coords
            #get indices where coordinates are not [0.,0.,0.]
            #sum across row for accurate results, there may be a more efficient way?
            #idx = torch.tensor([p for p,co in enumerate(y[i]) if co.sum() != 0], dtype=torch.long)
            idx = torch.tensor(mask[i][mask[i] != 0], dtype=torch.long)
            #print(idx.size())
            xdist_mat = pair_dist(torch.index_select(x[i], 0, idx))
            ydist_mat = pair_dist(torch.index_select(y[i], 0, idx))
            
            D = ydist_mat - xdist_mat
            dRMSD += torch.norm(D, 2)/((idx.size(0)**2 - idx.size(0))**0.5)
            
        return dRMSD/x.size(0) #average over the batch
    
def create_emb_layer(aa2vec_path):
    aa2vec = torch.tensor(bcolz.open(aa2vec_path), requires_grad=True)
    vocab_sz, embed_dim = aa2vec.size()
    emb_layer = nn.Embedding(vocab_sz, embed_dim)
    emb_layer.load_state_dict({'weight': aa2vec})

    return emb_layer, vocab_sz, embed_dim
