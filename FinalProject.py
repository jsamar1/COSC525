from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

experiment_id = "5rRNlAn0SpaKi9hOinW9xg"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
runs = {x: y[df.tag=='validation_loss'] for x, y in df.groupby('run', as_index=False)}
time = {'AvgPool': 39,'MaxPool': 38, 'Area': 37.5, 'Nearest-Exact': 38.5, 'Pruned Random': 34, 'Pretrained': 37, 'Full Random': 37, 'Full Weight Transfer': 41}
timedf = pd.DataFrame(columns=time.keys())
for name,val in time.items():
    timedf[name]=val
models = []
x = runs['AvgPool']['step'].to_numpy()

plt.plot(x,runs['AvgPool']['value'])
plt.plot(x,runs['MaxPool']['value'])
plt.plot(x,runs['InterpNearestExact']['value'])
plt.plot(x,runs['InterpArea']['value'])
plt.plot(x,runs['PrunedModelRandomInit']['value'])
plt.legend(['AvgPool','MaxPool','InterpNearestExact','InterpArea','Random Initialization'])
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.savefig('PrunedSize.png')
plt.show()

plt.plot(x,runs['PretrainedModel']['value'])
plt.plot(x,runs['FullModelWeightTransfer']['value'])
plt.plot(x,runs['FullModelRandomInit']['value'])
plt.legend(['Pretrained','Weight Transfer','Random Initalization'])
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.savefig('FullSize.png')
plt.show()

#plt.bar(range(len(time)),list(time.values()))
plt.figure(figsize=(12, 3))  # width:20, height:3
plt.bar(range(len(time)), time.values(), align='center', width=0.5)
plt.ylabel('Minutes to Train')
plt.xticks(range(len(time)),list(time.keys()),rotation=0)
plt.savefig('TimeToTrain.png')
plt.show()