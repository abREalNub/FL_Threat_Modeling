from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from flex.pool import init_server_model
from flex.pool import FlexPool
from flex.model import FlexModel
from flex.data.lazy_indexable import LazyIndexable

from networks_execution import *
from networks_models import *
from process_data import *

import numpy as np


def label_flipping_attack(client_labels, num_labels):
    """
    Data poisoning attack which changes the labels of the training data on the malicious clients.
    Implementación basada en: 1- Gehlhar, T., Marx, F., Schneider,
T., Suresh, A., Wehrle, T., &
Yalame, H. (2023, May). SafeFL:
MPC-friendly framework for private
and robust federated learning. In
2023 IEEE Security and Privacy
Workshops (SPW) (pp. 69-76).
IEEE.
    client_labels: ndarray. A client labels
    num_labels: integer. Highest label number
    """
    posion_label = deepcopy(client_labels)
    for label_ind in range(len(posion_label)):
        posion_label[label_ind] = num_labels - client_labels[label_ind] - 1

    return posion_label