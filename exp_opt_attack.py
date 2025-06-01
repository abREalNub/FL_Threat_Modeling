from copy import deepcopy
import copy
import torch.cuda
from flex.pool import init_server_model, deploy_server_model, aggregate_weights
from flex.pool import FlexPool
from flex.model import FlexModel

from process_data import *
from networks_execution import *
from flexclash.model import model_poison_agregator
from flexclash.pool import median, bulyan, trimmed_mean, multikrum

from attack.agr_agm_attack import agr_agnostic_attack
from attack import param_manipulated_attacks as param_manipulated
from flex.pool import set_aggregated_diff_weights_pt
from flex.pool import collect_client_diff_weights_pt
from flex.pool import fed_avg

import random as ran

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


# Def attack to model aggregator

@model_poison_agregator
def opt_attack(aggregated_weights_as_a_list: list):
    """
    Implementación del ataque AGR_Agnostic_Attack dentro del flujo de Flex.
    """
    # Extraer configuración general del ataque

    num_corrupt = 2  # Ajustar según el número de clientes maliciosos en el experimento
    num_agents = len(clients.actor_ids)  # Número total de clientes en el FL
    print("Antes de la modif", torch.sum(aggregated_weights_as_a_list[0][0]))

    mod_weights = param_manipulated.adecuate_params(aggregated_weights_as_a_list)
    # Diccionario de actualizaciones de los clientes
    agent_updates_dict = mod_weights

    # Lista de identificadores de los clientes seleccionados
    agent_id_id = selected_test_clients.actor_ids


    # Calcular el promedio de actualizaciones (avg_updates) de los clientes benignos
    benign_clients = ran.sample(agent_id_id, len(agent_id_id) - num_corrupt)



    avg_updates = sum([agent_updates_dict[client_id] for client_id in benign_clients]) / len(benign_clients)

    # Seleccionar un cliente malicioso arbitrario (ajustar según el diseño del experimento)
    i = num_corrupt

    # Definir el umbral de distancia
    dis = 0.5  # Ajustar según la tolerancia del ataque

    # Ejecutar el ataque
    agent_updates_dict, max_dis_poi = agr_agnostic_attack(num_corrupt, num_agents, agent_updates_dict, agent_id_id,
                                                          avg_updates, i, dis)

    mod_weights_original = param_manipulated.adecuate_params_reverse(agent_updates_dict, aggregated_weights_as_a_list[0])

    print("Después de la modif", torch.sum(agent_updates_dict[0][0]))

    return mod_weights_original


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
        # print("Aggregated weights:", flex_pool.aggregators.map(collect_client_diff_weights_pt, selected_clients))
        flex_pool.aggregators.map(opt_attack)
        # print("Aggregated weights test:", agg_weights)
        flex_pool.aggregators.map(median)

        # The aggregator send its aggregated weights to the server
        flex_pool.aggregators.map(set_aggregated_diff_weights_pt, flex_pool.servers)
        metrics = flex_pool.servers.map(evaluate_global_model)
        loss, acc = metrics[0]
        print(f"Global accuracy Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")

        # Optional
        selected_clients.map(clean_up)


if __name__ == "__main__":
    train_n_rounds(1, 10)
