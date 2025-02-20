from copy import deepcopy
from flex.data.lazy_indexable import LazyIndexable
from flex.data import Dataset
import functools
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from flex.model import FlexModel

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

def generate_bad_data_for_test(func):
    
    @functools.wraps(func)
    def _poison_Testset_(
        Dataset_test: Dataset,
        *args, 
        **kwargs
    ):
        def poison_data_for_test(test_dataset,*args, **kwargs):
            try:                
                new_features, new_labels  = func(test_dataset, *args, **kwargs)
            except ValueError:
                raise ValueError(
                        f"The decorated function: {func.__name__} must return two values: the new features and labels."
                )
            return new_features, new_labels
        
        def execute_test_poison():
            new_features, new_labels = poison_data_for_test(deepcopy(Dataset_test), *args, **kwargs)
            return Dataset(X_data = LazyIndexable(new_features, length=len(new_features)), 
                            y_data = LazyIndexable(new_labels, length=len(new_labels)))
        
        return execute_test_poison()

    return _poison_Testset_


def evaluate_model_with_poison_data(func):

    @functools.wraps(func)
    def _poison_all_Testset_(
        server_model: FlexModel,
        poison_data: Dataset,
        *args, 
        **kwargs
    ):
        def poison_data_for_test(server_model, poison_data,*args, **kwargs):
            try:              
                test_loss, test_acc = func(server_model, poison_data, *args, **kwargs)
            except ValueError:
                raise ValueError(
                        f"The decorated function: {func.__name__} must return two values: the test loss and test accuracy."
                )
            return test_loss, test_acc
        
        return poison_data_for_test(server_model, poison_data, *args, **kwargs) 

    return _poison_all_Testset_


def data_poison_evaluator_pt(server_model, poison_dataset):
    model = server_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_model["criterion"]
    test_dataloader = DataLoader(
        poison_dataset, batch_size=256, shuffle=True, pin_memory=False
    )
    losses = []
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target.long()).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc