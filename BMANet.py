from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from utils import reset, uniform, zeros
from torch.nn import Sequential, Linear, ReLU


class AMPConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 edge_inchannels: int, hidden_channels: int,
                 aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super(AMPConv, self).__init__(aggr=aggr, **kwargs)

        self.e_nn = torch.nn.Linear(edge_inchannels,hidden_channels)
        self.n_nn = torch.nn.Linear(in_channels,hidden_channels)
        self.nn = torch.nn.Linear(hidden_channels,out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels_l = in_channels[0]

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        reset(self.n_nn)
        reset(self.e_nn)
        if self.root is not None:
            uniform(self.root.size(0), self.root)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None and self.root is not None:
            out += torch.matmul(x_r, self.root)

        if self.bias is not None:
            out += self.bias
        out = F.relu(out)

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        hidden_message = F.relu(self.e_nn(edge_attr)) + F.relu(self.n_nn(x_j))
        hidden_message = F.relu(self.nn(hidden_message))
        return hidden_message

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class BMA_update(torch.nn.Module):
    def __init__(self,node_channels,edge_channels,hidden_channels,out_channels):
        super(BMA_update, self).__init__()
        self.n_nn = Linear(node_channels*2,hidden_channels)
        self.root = Linear(edge_channels,hidden_channels)
        self.nn = Linear(hidden_channels,out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.n_nn)
        reset(self.nn)
        reset(self.root)
    def forward(self,x, edge_index, edge_attr):
        hidden_message = F.relu(self.n_nn(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1))) + F.relu(self.root(edge_attr))
        return F.relu(self.nn(hidden_message))


class BMANet(torch.nn.Module):
    def __init__(self):
        super(BMANet, self).__init__()
        self.layer1 = ENConv(in_channels=9,out_channels=64,edge_inchannels=3,hidden_channels=128)
        self.eu1 = Edge_update(node_channels=64,edge_channels=3,hidden_channels=128,out_channels=64)
        self.att_node1 = Sequential(Linear(64,64*4),ReLU(),Linear(64*4,64))
        self.att_edge1 = Sequential(Linear(64,64*4),ReLU(),Linear(64*4,64))

        self.layer2 = ENConv(in_channels=64,out_channels=64,edge_inchannels=64,hidden_channels=512)
        self.eu2 = Edge_update(node_channels=64,edge_channels=64,hidden_channels=128,out_channels=64)
        self.att_node2 = Sequential(Linear(64,64*4),ReLU(),Linear(64*4,64))
        self.att_edge2 = Sequential(Linear(64,64*4),ReLU(),Linear(64*4,64))

        self.layer3 = ENConv(in_channels=64,out_channels=64,edge_inchannels=64,hidden_channels=512)
        self.eu3 = Edge_update(node_channels=64,edge_channels=64,hidden_channels=128,out_channels=64)
        self.att_node3 = Sequential(Linear(64,64*4),ReLU(),Linear(64*4,64))
        self.att_edge3 = Sequential(Linear(64,64*4),ReLU(),Linear(64*4,64))

        self.layer4 = ENConv(in_channels=64, out_channels=64, edge_inchannels=64, hidden_channels=512)
        self.eu4 = Edge_update(node_channels=64, edge_channels=64, hidden_channels=128, out_channels=64)
        self.att_node4 = Sequential(Linear(64,64*4),ReLU(),Linear(64*4,64))
        self.att_edge4 = Sequential(Linear(64,64*4),ReLU(),Linear(64*4,64))

        self.set2set = Set2Set(64, processing_steps=3)
        self.att = Sequential(Linear(64*8,64*8),ReLU(),Linear(64*8,64*8))
        self.lin1 = Linear(64*8,64*2)
        self.lin2 = Linear(64*2,2)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.lin1)
        reset(self.att)
        reset(self.lin2)
        reset(self.att_edge1)
        reset(self.att_node1)
        reset(self.att_edge2)
        reset(self.att_node2)
        reset(self.att_edge3)
        reset(self.att_node3)
        reset(self.att_edge4)
        reset(self.att_node4)


    def forward(self,data):
        x, edge_index, edge_attr, x_batch,edge_attr_batch = data.x, data.edge_index, data.edge_attr, data.x_batch,data.edge_attr_batch
        x = self.layer1(x,edge_index,edge_attr)
        edge_attr = self.eu1(x,edge_index,edge_attr)
        x = torch.mul(F.softmax(self.att_node1(x), dim=1), x) * x.size(1)
        edge_attr = torch.mul(F.softmax(self.att_edge1(edge_attr), dim=1), edge_attr) * edge_attr.size(1)

        x = self.layer2(x,edge_index,edge_attr)
        edge_attr = self.eu2(x,edge_index,edge_attr)
        x = torch.mul(F.softmax(self.att_node2(x), dim=1), x) * x.size(1)
        edge_attr2 = torch.mul(F.softmax(self.att_edge2(edge_attr), dim=1), edge_attr) * edge_attr.size(1)

        x = self.layer3(x,edge_index,edge_attr2)
        edge_attr = self.eu3(x,edge_index,edge_attr2)
        x = torch.mul(F.softmax(self.att_node3(x), dim=1), x) * x.size(1)
        edge_attr = torch.mul(F.softmax(self.att_edge3(edge_attr), dim=1), edge_attr) * edge_attr.size(1)

        x = self.layer4(x,edge_index,edge_attr)
        edge_attr = self.eu4(x,edge_index,edge_attr)


        x_out = self.set2set(x, x_batch)
        edge_attr_out = self.set2set(edge_attr,edge_attr_batch)
        pool_node = torch.cat([gmp(x, x_batch),
                            gap(x, x_batch)], dim=1)
        pool_edge = torch.cat([gmp(edge_attr,edge_attr_batch),gap(edge_attr,edge_attr_batch)],dim = 1)
        out = torch.cat([x_out,edge_attr_out,pool_node,pool_edge],dim = 1)
        out = torch.mul(F.softmax(self.att(out), dim=1), out) * out.size(1)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out
