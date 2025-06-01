from copy import deepcopy
import copy
import torch.cuda
from flex.pool import init_server_model, deploy_server_model, aggregate_weights
from flex.pool import FlexPool
from flex.model import FlexModel

from process_data import *
from networks_execution import *
from flexclash.model import model_poison_agregator
from flexclash.pool import median, bulyan, trimmed_mean, multikrum, central_differential_privacy


from attack import param_manipulated_attacks as param_manipulated
from flex.pool import set_aggregated_diff_weights_pt
from flex.pool import collect_client_diff_weights_pt


import random as ran
from attack.DeSMP_variants import desmp_attack_1, desmp_attack

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.cuda.is_available()
    else "cpu"
)

flex_dataset, server_id = load_and_preprocess_horizontal(dataname="mnist", trasnform=False, nodes=10)

net_config = ExecutionNetwork()


@init_server_model
def build_server_model():
    server_flex_model = FlexModel()
    criterion, model, optimizer = net_config.for_fd_server_model_config()
    server_flex_model["model"] = model.to(device)
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = criterion
    server_flex_model["optimizer_func"] = optimizer
    server_flex_model["optimizer_kwargs"] = {}
    return server_flex_model


# Model arquitecture config

flex_pool = FlexPool.client_server_pool(
    fed_dataset=flex_dataset, server_id=server_id, init_func=build_server_model
)

clients = flex_pool.clients
servers = flex_pool.servers
aggregators = flex_pool.aggregators

print(f"Number of nodes in the pool {len(flex_pool)}: {len(servers)} server plus {len(clients)} clients. The server is "
      f"also an aggregator")

# Select clients
clients_per_round = 10
selected_test_clients_pool = clients.select(clients_per_round)
selected_test_clients = selected_test_clients_pool.clients

print(f'Server node is identified by key "{servers.actor_ids[0]}"')
print(
    f"Selected {len(selected_test_clients.actor_ids)} client nodes of a total of {len(clients.actor_ids)}"
)


@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    return copy.deepcopy(server_flex_model)


def train(client_flex_model: FlexModel, client_data: Dataset):
    print(np.array(client_data.X_data).shape)
    train_dataset = client_data.to_torchvision_dataset(transform=mnist_transform())
    client_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = client_flex_model['model']
    model = model.to(device)

    client_flex_model["previous_model"] = deepcopy(
        model
    )
    optimizer = client_flex_model["optimizer_func"]
    criterion = client_flex_model["criterion"]

    net_config.train_network(local_epochs=1, criterion=criterion, optimizer=optimizer, momentum=0.9, lr=0.005,
                             trainloader=client_dataloader, testloader=None,
                             model=model)

    return client_flex_model


@model_poison_agregator
def desmp(list_of_weights: list):
    # Extraer configuración del ataque
    sigma = 0.1  # Ajustar según el nivel de DP en el experimento
    ganma = 0.5

    # Transformar pesos
    mod_dic = param_manipulated.adecuate_params(list_of_weights)

    sensitivity = torch.median(torch.stack([torch.norm(param) for param in mod_dic]))



    # Lista de clientes participantes
    agent_id_id = selected_test_clients.actor_ids

    # Definir clientes maliciosos
    num_corrupt = ran.sample(agent_id_id, 1)  # Simulación con un cliente malicioso

    # Aplicar ataque DeSMP
    attacked_updates = desmp_attack(mod_dic, sigma, sensitivity, ganma, num_corrupt, agent_id_id)

    # Revertir transformacion
    final_upd = param_manipulated.adecuate_params_reverse(attacked_updates, list_of_weights[0])

    return final_upd


# Aggregate to FL model (step to complete)

# Aggregate weights(step to complete)

# Eval global Model

def evaluate_global_model(server_flex_model: FlexModel, test_data: Dataset):
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)

    criterion = server_flex_model["criterion"]
    # get test data as a torchvision object
    test_dataset = test_data.to_torchvision_dataset(transform=mnist_transform())
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=False
    )
    losses = []
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count

    return test_loss, test_acc


# Cleaning

def clean_up(client_model: FlexModel, _):
    import gc
    client_model.clear()
    gc.collect()


# Summing up

def train_n_rounds(n_rounds=2, clients_per_round=10):
    for i in range(n_rounds):
        print(f"\nRunning round: {i + 1} of {n_rounds}")
        selected_clients_pool = flex_pool.clients.select(clients_per_round)
        selected_clients = selected_clients_pool.clients
        print("Selected clients:", len(selected_clients))
        print(f"Selected clients for this round: {len(selected_clients)}")
        # Deploy the server model to the selected clients
        flex_pool.servers.map(copy_server_model_to_clients, selected_clients)
        # Each selected client trains her model
        selected_clients.map(train)
        # The aggregador collects weights from the selected clients and aggregates them
        flex_pool.aggregators.map(collect_client_diff_weights_pt, selected_clients)
        flex_pool.aggregators.map(desmp)
        flex_pool.aggregators.map(central_differential_privacy)

        # The aggregator send its aggregated weights to the server
        flex_pool.aggregators.map(set_aggregated_diff_weights_pt, flex_pool.servers)
        metrics = flex_pool.servers.map(evaluate_global_model)
        loss, acc = metrics[0]
        print(f"Global accuracy Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")

        # Optional
        selected_clients.map(clean_up)


if __name__ == "__main__":
    train_n_rounds(1, 10)
