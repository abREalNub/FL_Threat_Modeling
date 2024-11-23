import copy

import torch.cuda
from flex.data import Dataset
from flex.datasets import load
from flex.pool import init_server_model, deploy_server_model, collect_clients_weights, set_aggregated_weights, \
    aggregate_weights
from flex.pool import FlexPool
from flex.model import FlexModel

import tensorly as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models import CNN
from models import nets

device = ("cuda"
          if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

flex_dataset, test_data = load("federated_emnist", return_test=True, split="digits")

# Assign test data to server_id
server_id = "server"
flex_dataset[server_id] = test_data

mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


@init_server_model
def build_server_model():
    server_flex_model = FlexModel()

    server_flex_model["model"] = CNN(10)
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}
    return server_flex_model


flex_pool = FlexPool.client_server_pool(
    flex_dataset, server_id=server_id, init_func=build_server_model
)

clients = flex_pool.clients
servers = flex_pool.servers
aggregators = flex_pool.aggregators

print(f"Number of nodes in the pool{len(flex_pool)}: {len(servers)} server plus {len(clients)} clients. The server is "
      f"also an aggregator")

# Select clients
clients_per_round = 10
selected_clients_pool = clients.select(clients_per_round)
selected_clients = selected_clients_pool.clients

print(f'Server node is indentified by key "{servers.actor_ids[0]}"')
print(
    f"Selected {len(selected_clients.actor_ids)} client nodes of a total of {len(clients.actor_ids)}"
)

@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    return copy.deepcopy(server_flex_model)


servers.map(copy_server_model_to_clients, selected_clients)


def train(client_flex_model: FlexModel, client_data: Dataset):
    train_dataset = client_data.to_torchvision_dataset(transform=mnist_transforms)
    client_dataloader = DataLoader(train_dataset, batch_size=20)
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    for _ in range(5):
        for imgs, labels in client_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()


selected_clients.map(train)


@collect_clients_weights
def get_clients_weights(client_flex_model: FlexModel):
    weight_dict = client_flex_model["model"].state_dict()
    return [weight_dict[name] for name in weight_dict]


aggregators.map(get_clients_weights, selected_clients)

tl.set_backend("pytorch")


@aggregate_weights
def aggregate_with_fedavg(list_of_weights: list):
    agg_weights = []
    for layer_index in range(len(list_of_weights[0])):
        weights_per_layer = [weights[layer_index] for weights in list_of_weights]
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.mean(weights_per_layer, axis=0)
        agg_weights.append(agg_layer)
    return agg_weights


# Aggregate weights
aggregators.map(aggregate_with_fedavg)


@set_aggregated_weights
def set_agreggated_weights_to_server(server_flex_model: FlexModel, aggregated_weights):
    with torch.no_grad():
        weight_dict = server_flex_model["model"].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            weight_dict[layer_key].copy_(new)


aggregators.map(set_agreggated_weights_to_server, servers)


def evaluate_global_model(server_flex_model: FlexModel, test_data: Dataset):
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    # get test data as a torchvision object
    test_dataset = test_data.to_torchvision_dataset(transform=mnist_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, shuffle=True, pin_memory=False
    )
    losses = []
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc


metrics = servers.map(evaluate_global_model)
print(metrics[0])

#Summing up
"""
def train_n_rounds(n_rounds, clients_per_round=20):
    pool = FlexPool.client_server_pool(
        fed_dataset=flex_dataset, server_id=server_id, init_func=build_server_model
    )
    for i in range(n_rounds):
        print(f"\nRunning round: {i+1} of {n_rounds}")
        selected_clients_pool = pool.clients.select(clients_per_round)
        selected_clients = selected_clients_pool.clients
        print(f"Selected clients for this round: {len(selected_clients)}")
        # Deploy the server model to the selected clients
        pool.servers.map(copy_server_model_to_clients, selected_clients)
        # Each selected client trains her model
        selected_clients.map(train)
        # The aggregador collects weights from the selected clients and aggregates them
        pool.aggregators.map(get_clients_weights, selected_clients)
        pool.aggregators.map(aggregate_with_fedavg)
        # The aggregator send its aggregated weights to the server
        pool.aggregators.map(set_agreggated_weights_to_server, pool.servers)
        metrics = pool.servers.map(evaluate_global_model)
        loss, acc = metrics[0]
        print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
"""