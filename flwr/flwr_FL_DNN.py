from collections import OrderedDict
from typing import List, Tuple
import sys
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common import Metrics
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
import pathlib
import random

from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from flwr.common import Metrics
from torch.utils.data import DataLoader, random_split

from logging import WARNING
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from typing import Callable, Dict, List, Optional, Tuple, Union

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from functools import reduce

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average_fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    loss_all = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(loss_all) / sum(examples)}


class strategy_custom(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        topology: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.topology = topology
        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        # Add additional arguments below
        self.parameters_for_client = []
        self.ids_for_client = []

    def Laplacian_Matrix(self, NUM_AGENTS, topology='Ring'):
        if topology == 'No':
            L = np.eye(NUM_AGENTS)

        if topology == 'Ring':
            L = 0.5 * np.eye(NUM_AGENTS) + 0.25 * np.eye(NUM_AGENTS, k=1) + 0.25 * np.eye(NUM_AGENTS,
                                                                                          k=-1) + 0.25 * np.eye(
                NUM_AGENTS, k=NUM_AGENTS - 1) + 0.25 * np.eye(NUM_AGENTS, k=-NUM_AGENTS + 1)

        if topology == 'Full':
            A = np.ones([NUM_AGENTS, NUM_AGENTS]) - np.eye(NUM_AGENTS)
            L = (A + sum(A[0]) * np.eye(NUM_AGENTS)) / sum(A + sum(A[0]) * np.eye(NUM_AGENTS))

        if topology == 'MS':
            A = np.random.randint(2, size=NUM_AGENTS * NUM_AGENTS)
            A = (np.ones([NUM_AGENTS, NUM_AGENTS]) - np.eye(NUM_AGENTS)) * A.reshape([NUM_AGENTS, NUM_AGENTS])
            vec = A + np.diag(A.sum(axis=1))
            zero_id = np.where(vec.sum(axis=1) == 0)
            for k in range(len(zero_id[0])):
                vec[zero_id[0][k]][zero_id[0][k]] = 1
            L = vec / vec.sum(axis=1).reshape(-1, 1)
        return L
        
    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Define (TMP) Laplace Matrix
        L_support = self.Laplacian_Matrix(len(results),topology=self.topology)
        # I am not sure the weights update law here is right or not. My update law would be X = np.matmul(L,X).
        # With our update law, After several steps, the value of X would converge to a single value.


        # L_support = np.zeros_like(L_support)

        # Create a list of weights, each multiplied by the related number of examples
        # weighted_weights = [
        #     [layer * num_examples for layer in weights] for weights, num_examples in results
        # ]
        weighted_weights = []
        for weights, num_examples in results:
            support = []
            for layer in weights:
                # support.append(layer * num_examples)
                support.append(layer)
            weighted_weights.append(support)

        # Compute average weights of each layer
        # weights_prime: NDArrays = [
        #     reduce(np.add, layer_updates) / num_examples_total
        #     for layer_updates in zip(*weighted_weights)
        # ]
        weights_prime_list = []
        for message_id in range(len(results)):
            weights_prime: NDArrays = []
            for layer_updates in zip(*weighted_weights):
                # print(len(layer_updates)) # number of clients
                accumulate_sum = np.zeros_like(layer_updates[0])
                for i in range(len(layer_updates)):
                    accumulate_sum = np.add(accumulate_sum, L_support[message_id, i] * layer_updates[i])
                # support = reduce(np.add, layer_updates) / num_examples_total
                # Do we need to divide by num_examples_total if in dl scheme?
                # support = accumulate_sum / num_examples_total
                support = accumulate_sum
                weights_prime.append(support)
            weights_prime_list.append(weights_prime)

        self.parameters_for_client = weights_prime_list

        # Temporarily choose 0 for global model update, need discuss this later
        return weights_prime_list[0]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # print(results[0][0], results[1][0]) # <flwr.server.grpc_server.grpc_client_proxy.GrpcClientProxy object at 0x000002548A0FFB48>
        list_client_id = []
        for i in range(len(results)):
            list_client_id.append(results[i][0])
        self.ids_for_client = list_client_id
        # print(len(weights_results)) # 2: two client have weight
        # print(len(weights_results[0])) # 2: 0 indicates parameters, 1 indicates num_examples
        # print(len(weights_results[0][0])) # 16: for each parameter set, 16 layers have parameters
        # print(weights_results[0][0][0].shape)


        # with open(f'Results/test_round{server_round}.txt', 'w') as outfile:
        #     for slice_4d in weights_results:
        #         for slice_3d in slice_4d[0]:
        #             # for slice_2d in slice_3d:
        #             np.savetxt(outfile, slice_3d)
        for i in range(len(weights_results)):
            with open(f'Results/client_{i}_round{server_round}.txt', 'w') as outfile:
            # for slice_4d in weights_results:
                for slice_3d in weights_results[i][0]:
                    # for slice_2d in slice_3d:
                    np.savetxt(outfile, slice_3d)


        parameters_aggregated = ndarrays_to_parameters(self.aggregate(weights_results))

        # Post process parameter list
        if len(self.parameters_for_client) != 0:
            for i in range(len(self.parameters_for_client)):
                self.parameters_for_client[i] = ndarrays_to_parameters(self.parameters_for_client[i])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        
        return parameters_aggregated, metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        fit_ins_default = FitIns(parameters, config)

        # Post process FitIns for each client
        fit_ins_foreach_client = []
        # print(len(self.parameters_for_client))
        if len(self.parameters_for_client) != 0:
            for i in range(len(self.parameters_for_client)):
                fit_ins_foreach_client.append(FitIns(self.parameters_for_client[i], config))
        else:
            # default global parameter configuration
            fit_ins_foreach_client.append(fit_ins)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # clients = client_manager.sample(
        #     num_clients=sample_size, min_num_clients=min_num_clients
        # )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=sample_size
        )

        # Return client/config pairs
        client_pairs = []
        # for client in clients:
        #     client_pairs.append((client, fit_ins))
        # return [(client, fit_ins) for client in clients]

        # for client in clients:
        #     client_pairs.append((client, fit_ins))
        
        client_table = [0 for _ in clients]
        for idx_enumerator, client in enumerate(clients):
            flag = 0
            for client_idx in range(len(self.ids_for_client)):
                # if found the corresponding client in the parameter list
                if client == self.ids_for_client[client_idx]:
                    client_pairs.append((client, fit_ins_foreach_client[client_idx]))
                    client_table[idx_enumerator] = 1
                    flag = 1
                    # print('found:{}'.format(client))
                    break
            if flag == 0:
                client_pairs.append((client, fit_ins_default))
            
        # if len(self.ids_for_client) != 0:
        #     print(self.ids_for_client[0] == clients[0])
        #     print(self.ids_for_client[0] == clients[1])
        # print(self.ids_for_client)
        self.parameters_for_client = []
        self.ids_for_client = []
        return client_pairs


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
torch.manual_seed(10)
random.seed(10)
np.random.seed(seed=10)
# parameters
time_step = 48
BATCH_SIZE = 32
data = []
with open (pathlib.Path(__file__).parent.absolute()/'House_30.txt', 'r') as reader:
  for line in reader:
    stripped_line = line.strip().split()
    data.append(stripped_line)

tem = [x[0] for x in data]
houses = list(sorted(set(tem)))

date = []
consumption = []
for i in houses:
  date.append([float(x[1]) for x in data if x[0]==i])
  consumption.append([float(x[2]) for x in data if x[0]==i]) 

def create_label(data, time_step):
  x_nest, y_nest = [], []
  for j in range(len(data)):
    x_data, y_data = [], []
    for i in range(len(data[j]) - time_step):
      x = data[j][i: (i + time_step)]
      x_data.append(x)
      y = [data[j][i + time_step]]
      y_data.append(y)

    #x_data = np.array(x_data)[:, :, np.newaxis]
    x_data = np.array(x_data)[:, :]
    #x_data = np.array(x_data)[:, np.newaxis, :]
    x_nest.append(x_data)
    y_nest.append(y_data)
  x_nest = np.array(x_nest)
  y_nest = np.array(y_nest)
  return x_nest, y_nest
# 可能要去掉x的最后一个维度 从（48，1）变为（48）
input, labels = create_label(consumption, time_step)
input = np.float32(input)
labels = np.float32(labels)


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

length = len(input[0])
val = int(0.7*length)
test = int(0.9*length)
trainloaders = []
valloaders = []
testloaders = []

def load_datasets(input, labels):

  Xtrain_raw = [x[0: val] for x in input]
  Xval_raw = [x[val: test] for x in input]
  Xtest_raw = [x[test: ] for x in input]

  Ytrain_raw = [x[0: val] for x in labels]
  Yval_raw = [x[val: test] for x in labels]
  Ytest_raw = [x[test: ] for x in labels]

  for i in range(30):
    ds_train = GetLoader(Xtrain_raw[i], Ytrain_raw[i])
    trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False))
    ds_val = GetLoader(Xval_raw[i], Yval_raw[i])
    valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    ds_test= GetLoader(Xtest_raw[i], Ytest_raw[i])
    testloaders.append(DataLoader(ds_test, batch_size=BATCH_SIZE))

  return trainloaders, valloaders, testloaders
trainloaders, valloaders, testloaders = load_datasets(input, labels)

MODEL = 'DNN'
# MODEL = 'WAVENET'
# MODEL = 'CNN'
# MODEL = 'LSTM'

STRATEGY = 'FL'
SUBSTRATEGY = 'None'

STRATEGY = 'DL'
SUBSTRATEGY = 'Ring'
# SUBSTRATEGY = 'Full'
# SUBSTRATEGY = 'MS'




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Linear(48, 48),
            act(),
            nn.Linear(48, 48),
            act(),
            nn.Linear(48, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        return out

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, 2)
        self.act = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(32*47*6,64)
        self.fc2 = nn.Linear(64,1)

    def forward(self,x):
        x = x.view(32, 1, 48)
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.act(x)
        out = self.fc2(x)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1, batch_size=32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(DEVICE)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(DEVICE)
        seq_len = input_seq.shape[1]
        input_seq = input_seq.view(self.batch_size, seq_len, 1)
        
        output, _ = self.lstm(input_seq, (h_0, c_0)) # output(32, 48, 32)
        output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)  # (32 * 48, 32)


        pred = self.linear(output)  # (32*48, 1)
        pred = pred.view(self.batch_size, seq_len, -1) #(32,48,1)
        pred = pred[:, -1, :]  # (32, 1)
        return pred 
if MODEL == 'DNN':
    net = Net().to(DEVICE)

elif MODEL=='CNN':   
    net = CNN().to(DEVICE)

elif MODEL == 'LSTM':
    net = LSTM().to(DEVICE)

elif MODEL == 'WAVENET':
    sys.exit('NOT FINISH YET, PLEASE TRY OTHER MODELS.')
    
else:
    sys.exit('NOT SUPPORT MODEL TYPE IN THIS VERSION, PLEASE TRY MODELS FROM DNN, CNN, LSTM, and WAVENET.')




























criterion = nn.MSELoss() 

def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
           
        epoch_loss /= len(trainloader.dataset)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}") 


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = net(x)
            loss += criterion(outputs, y).item()

    loss /= len(testloader.dataset)
   
    return loss 


# # Central part

# x, y = next(iter(trainloaders[0]))
# trainloader = trainloaders[0]
# valloader = valloaders[0]
# testloader = testloaders[0]
# net = Net().to(DEVICE)


# for epoch in range(5):
#     train(net, trainloader, 1)
#     loss = test(net, valloader)
#     print(f"Epoch {epoch+1}: validation loss {loss}")

# loss = test(net, testloader)
# print(f"Final test set performance:\n\tloss {loss}")



# FL part

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(0)}

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    # net = Net().to(DEVICE)
    if MODEL == 'DNN':
        net = Net().to(DEVICE)

    elif MODEL=='CNN':   
        net = CNN().to(DEVICE)

    elif MODEL == 'LSTM':
        net = LSTM().to(DEVICE)

    elif MODEL == 'WAVENET':
        sys.exit('NOT FINISH YET, PLEASE TRY OTHER MODELS.')
        
    else:
        sys.exit('NOT SUPPORT MODEL TYPE IN THIS VERSION, PLEASE TRY MODELS FROM DNN, CNN, LSTM, and WAVENET.')
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)

NUM_CLIENTS = 30

# Create FedAvg strategy




if STRATEGY == 'FL':

    strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=0.3,  # Sample 50% of available clients for evaluation
            min_fit_clients=30,  # Never sample less than 10 clients for training
            min_evaluate_clients=30,  # Never sample less than 5 clients for evaluation
            min_available_clients=30,  # Wait until all 10 clients are available
    )
else:
    strategy = strategy_custom(
        fraction_fit=1.0,
        fraction_evaluate=0.3,
        min_fit_clients=30,
        min_evaluate_clients=30,
        min_available_clients=30,
        evaluate_metrics_aggregation_fn = weighted_average,
        topology=SUBSTRATEGY,
        # fit_metrics_aggregation_fn = weighted_average_fit_metrics_aggregation_fn,
        )

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
DEVICE = torch.device("cpu") 
if DEVICE.type == "cuda":
  client_resources = {"num_gpus": 1}

# Start simulation
hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=20), #10
    strategy=strategy,
    client_resources=client_resources,
)

# np.save('Results/losses/loss_'+str(STRATEGY)+'_'+str(SUBSTRATEGY)+'_'+str(MODEL)+'_.npy',hist.losses_distributed)
