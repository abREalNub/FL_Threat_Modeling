import matplotlib.pyplot as plt
import numpy as np

functions = ['No attack','FedAvg',"Median",'Bulyan f=1 ','Bulyan f=2 ','Diferential Privacy']


data = {
    'No attack':[0.9711],
    'FedAvg':[0.9725],
    "Median":[0.9703],
    'Bulyan f=1':[0.9555],
    'Bulyan f=2':[0.9691],
    'Diferential Privacy':[0.9637]}

plt.figure(figsize=(10,6))


for func, values in data.items():
    plt.plot(func, pred_range, label=func)

plt.title("Distribution of")
plt.xlabel("Range")
plt.ylabel("Aggregators")
plt.legend()

plt.show()