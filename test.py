from process_data import *
from copy import deepcopy

from networks_models import *
from networks_execution import *
from flex.pool import init_server_model
from flex.model import FlexModel

from flex.pool import FlexPool
from flex.model import FlexModel
from flex.actors.actors import FlexActors
from flex.data import FedDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def verif_in_fr(list_fr, all_clients):
    for actor_ids in all_clients.actor_ids:
        if actor_ids in list_fr:
            return True
    return False


def util_for_fr(list_of_fr, all_clients):
    normal_clients = all_clients
    free_riding_clients = None

    if verif_in_fr(list_of_fr, all_clients):
        actors = FlexActors()
        models = {}
        new_data = FedDataset()

        actors_fr = FlexActors()
        models_fr = {}
        new_data_fr = FedDataset()

        def asing_client(place, id, from_the):
            place[id] = from_the[id]

        for actor_ids in all_clients.actor_ids:
            if actor_ids in list_of_fr:
                asing_client(actors_fr, actor_ids, all_clients._actors)
                asing_client(models_fr, actor_ids, all_clients._models)
                asing_client(new_data_fr, actor_ids, all_clients._data)
            elif actor_ids not in list_of_fr:
                asing_client(actors, actor_ids, all_clients._actors)
                asing_client(models, actor_ids, all_clients._models)
                asing_client(new_data, actor_ids, all_clients._data)
            new_pool_c = FlexPool(flex_actors=actors, flex_data=new_data, flex_models=models)
            new_pool_frc = FlexPool(flex_actors=actors_fr, flex_data=new_data_fr, flex_models=models_fr)
            normal_clients = new_pool_c.clients
            free_riding_clients = new_pool_frc.clients

    return normal_clients, free_riding_clients


def util_for_fr_join_all(clients_fr, clients, all_clients):
    clients_to_aggregate = []
    actors = FlexActors()
    models = {}
    new_data = FedDataset()

    def asing_client(place, id, from_the):
        place[id] = from_the[id]

    for actor_ids in all_clients.actor_ids:
        if actor_ids in clients_fr:
            asing_client(actors, actor_ids, clients_fr._actors)
            asing_client(models, actor_ids, clients_fr._models)
            asing_client(new_data, actor_ids, clients_fr._data)
        elif actor_ids not in clients_fr:
            asing_client(actors, actor_ids, clients._actors)
            asing_client(models, actor_ids, clients._models)
            asing_client(new_data, actor_ids, clients._data)
        clientss = FlexPool(flex_actors=actors, flex_data=new_data, flex_models=models)
        clients_to_aggregate = clientss.clients

    return clients_to_aggregate
