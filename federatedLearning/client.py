from flex.distributed import ClientBuilder
from flex.model import FlexModel
from flex.data import Dataset
import numpy as np
from typing import List

from federatedLearning.experiment import flex_dataset
import experiment

dataset = flex_dataset[list(flex_dataset.keys())[0]]

client = ClientBuilder()\
    .build_model(experiment.build_server_model)\
    .dataset(dataset)\
    .collect_weights(experiment.get_clients_weights)\
    .train(experiment.train(FlexModel(), dataset))\
    .eval(experiment.evaluate_global_model(FlexModel(), dataset),dataset)\
    .set_weights(experiment.set_agreggated_weights_to_server)\
    .build()

client.run(address="localhost:8080")
