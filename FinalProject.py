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