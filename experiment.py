from copy import deepcopy
import copy
import torch.cuda
from flex.pool import init_server_model, deploy_server_model, aggregate_weights
from flex.pool import FlexPool
from flex.model import FlexModel
from PIL import Image
from process_data import *
from networks_execution import *
from flexclash.data import data_poisoner_all
from flexclash.pool import central_differential_privacy, median, bulyan, trimmed_mean, multikrum
from poison_attack_evaluator import generate_bad_data_for_test, evaluate_model_with_poison_data, \
    data_poison_evaluator_pt
from attack import bad_net as bn
from attack.backdoorss import sniper_backdoor
import argparse
from flex.pool import set_aggregated_diff_weights_pt
from flex.pool import collect_client_diff_weights_pt
from flex.pool import fed_avg
import tensorly as tl

parser = argparse.ArgumentParser(description='Trying to reproduce the basic backdoor attack in "BadNets:___" into a '
                                             'federated learning model')
parser.add_argument('--data_path', default='.', help='Place to load dataset (default: .)')
# poisoning settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, '
                                                                      'default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, '
                                                                 'default: 0)')
parser.add_argument('--trigger_path', default="C:\\Users\\Adrian\\PycharmProjects\\pythonProject\\attack\\source"
                                              "\\trigger_white.png", help='Trigger Path (default: '
                                                                          'C:\\Users\\Adrian\\PycharmProjects'
                                                                          '\\pythonProject\\attack\\source'
                                                                          '\\trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

args = parser.parse_args()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
"""
cifar_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

training_data = datasets.CIFAR10(
    root=".", train=True, download=False, transform=None
)

test_data = datasets.CIFAR10(
    root=".", train=True, download=False, transform=None
)

config = FedDatasetConfig(seed=0)
config.replacement = False
config.n_nodes = 100

flex_dataset = FedDataDistribution.from_config(
    centralized_data=Dataset.from_torchvision_dataset(training_data), config= config
)

# Assign test data to server_id
server_id = "server"
flex_dataset[server_id] = Dataset.from_torchvision_dataset(test_data)
"""

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


# Momento para definir el ataque

client_ids = list(flex_dataset.keys())
clients_to_backdoor = client_ids[:1]
print(f"Modified clients: {clients_to_backdoor}")

target = 1
source = 2
percent_to_change = 0.2

clients_to_change = clients_to_backdoor

"""
@data_poisoner_all
def backdoor_bad_net(dataset_client: Dataset):
    new_img = []
    target_label = []
    poisoned_dataset = bn.MNISTPoison(args, args.data_path, train=True, download=False, transform=mnist_transform())
    for img, label in poisoned_dataset:
        new_img.append(img)
        target.append(label)

    new_img = np.array(new_img)
    target_label = np.array(target)
    new_img_final = [Image.fromarray(new_img[arr][0]) for arr in range(len(new_img))]

    return new_img_final, target
"""


@data_poisoner_all
def backdoor_bad_net(dataset_client: Dataset):
    x, y = dataset_client.to_numpy()
    x = np.expand_dims(x, axis=1)
    new_img = []
    target_label = []

    poisoned_dataset = bn.ImagePoison(images=x, targets=y, trigger_path=args.trigger_path,
                                      trigger_size=args.trigger_size, trigger_label=args.trigger_label,
                                      poisoning_rate=args.poisoning_rate, transform=mnist_transform())

    for img, label in poisoned_dataset:
        new_img.append(img)
        target_label.append(label)

    new_img = np.array(new_img)
    target_label = np.array(target_label)
    new_img_final = [Image.fromarray(new_img[arr][0]) for arr in range(len(new_img))]

    return new_img_final, target_label


@data_poisoner_all
def backdoor_sniper(dataset_client: Dataset):
    x, y = dataset_client.to_numpy()
    x = np.expand_dims(x, axis=1)

    new_img, new_target = sniper_backdoor(data=x, targets=y, source_label=source, target_label=target,
                                          epsilon=percent_to_change)

    new_img_final = [Image.fromarray(new_img[arr][0]) for arr in range(len(new_img))]

    return new_img_final, new_target


flex_data_modif = flex_dataset.apply(backdoor_sniper, node_ids=clients_to_backdoor)

# Model arquitecture config

flex_pool = FlexPool.client_server_pool(
    fed_dataset=flex_data_modif, server_id=server_id, init_func=build_server_model
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


# servers.map(copy_server_model_to_clients, selected_clients)


def train(client_flex_model: FlexModel, client_data: Dataset):
    print(np.array(client_data.X_data).shape)
    train_dataset = client_data.to_torchvision_dataset(transform=mnist_transform())
    client_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)

    model = client_flex_model['model']
    model = model.to(device)

    client_flex_model["previous_model"] = deepcopy(
        model
    )
    optimizer = client_flex_model["optimizer_func"]
    criterion = client_flex_model["criterion"]

    net_config.trainNetwork(local_epochs=1, criterion=criterion, optimizer=optimizer, momentum=0.9, lr=0.005,
                            trainloader=client_dataloader, testloader=None,
                            model=model)

    return client_flex_model


# selected_clients.map(train)


# Aggregate to FL model

# aggregators.map(collect_client_diff_weights_pt, selected_clients)

# Aggregate weights

@aggregate_weights
def aggregate_with_fedavg(list_of_weights: list):
    agg_weights = []
    for layer_index in range(len(list_of_weights[0])):
        weights_per_layer = [weights[layer_index] for weights in list_of_weights]
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.mean(weights_per_layer, axis=0)
        agg_weights.append(agg_layer)
    return agg_weights

# aggregators.map(fed_avg)

# aggregators.map(set_aggregated_diff_weights_pt, servers)


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


# metrics = servers.map(evaluate_global_model)
# print(metrics[0])

# Eval the attack
"""
Modificar con BadNets
"""


@generate_bad_data_for_test
def poison_test_set(test_set: Dataset):
    x, y = test_set.to_numpy()
    x = np.expand_dims(x, axis=1)
    new_img = []
    target_label = []

    poisoned_dataset = bn.ImagePoison(images=x, targets=y, trigger_path=args.trigger_path,
                                      trigger_size=args.trigger_size, trigger_label=args.trigger_label,
                                      poisoning_rate=1, transform=mnist_transform())

    for img, label in poisoned_dataset:
        new_img.append(img)
        target_label.append(label)

    new_img = np.array(new_img)
    target_label = np.array(target_label)
    new_img_final = [Image.fromarray(new_img[arr][0]) for arr in range(len(new_img))]

    return new_img_final, target_label


@generate_bad_data_for_test
def poison_test_set_2(test_set: Dataset):
    x, y = test_set.to_numpy()  # Transformar a una dimension más
    x = np.expand_dims(x, axis=1)
    new_data, new_tagets = sniper_backdoor(data=x, targets=y, source_label=source,
                                           target_label=target, epsilon=1)

    new_data = [Image.fromarray(new_data[arr][0]) for arr in range(
        len(new_data))]  # Para asegurar que sigan siendo imágenes evitando conflictos con transformaciones

    return new_data, new_tagets


@evaluate_model_with_poison_data
def evaluator_pt(server_model: FlexModel, test_data: Dataset):
    poison_dataset = poison_test_set_2(test_data)
    poison_dataset = poison_dataset.to_torchvision_dataset(transform=mnist_transform())
    test_loss, test_acc = data_poison_evaluator_pt(server_model, poison_dataset)

    return test_loss, test_acc


# metrics_for_bad_data = servers.map(evaluator_pt)

# loss_b, acc_b = metrics_for_bad_data[0]
# print(f"Server: Test acc: {acc_b:.4f}, test loss: {loss_b:.4f}")


# Cleaning

def clean_up(client_model: FlexModel, _):
    import gc
    client_model.clear()
    gc.collect()


# Summing up
"""
Modificar de acuerdo al ataque o hacer una copia del metodo
"""


def train_n_rounds(n_rounds, clients_per_round=10):

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
        #print("Aggregated weights:", flex_pool.aggregators.map(collect_client_diff_weights_pt, selected_clients))
        flex_pool.aggregators.map(fed_avg)
        # The aggregator send its aggregated weights to the server
        flex_pool.aggregators.map(set_aggregated_diff_weights_pt, flex_pool.servers)
        metrics = flex_pool.servers.map(evaluate_global_model)
        loss, acc = metrics[0]
        print(f"Global accuracy Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
        metrics_for_bad_data = flex_pool.servers.map(evaluator_pt)
        loss_b, acc_b = metrics_for_bad_data[0]
        print(f"Attack effectiveness Server: Test acc: {acc_b:.4f}, test loss: {loss_b:.4f}")
        # Optional
        selected_clients.map(clean_up)


if __name__ == '__main__':
    # model = build_server_model
    # servers.map(copy_server_model_to_clients, selected_clients)
    # selected_clients.map(train)
    # aggregators.map(get_clients_weights, selected_clients)
    # aggregators.map(aggregate_with_fedavg)
    # aggregators.map(set_agreggated_weights_to_server, servers)
    # metrics = servers.map(evaluate_global_model)
    # print(metrics[0])
    train_n_rounds(2, 2)
