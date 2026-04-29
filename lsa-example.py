import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from steps import (
    binary_fair_classification_step,
    predict_binary_classification_step,
    response_step,
)
from torchiteration import predict, train_plain as train, validate


config = {
    "dataset": "lsa",
    "training_step": "response_step",
    "batch_size": 16,
    "optimizer": "SGD",
    "optimizer_config": {},
    "scheduler": "StepLR",
    "scheduler_config": {
        "step_size": 2000,
        "gamma": 1,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "validation_step": "binary_fair_classification_step",
    "sensitive_index": 6,
}

model = torch.hub.load(
    "cat-claws/nn",
    "simplecnn",
    convs=[],
    linears=[7, 64, 32, 16, 8, 4],
    num_classes=1,
).to(config["device"])

with torch.no_grad():
    model.layers[1].weight.zero_()
    model.layers[1].weight += 1e-10

writer = SummaryWriter(
    comment=f"_{config['dataset']}_{model._get_name()}_{config['training_step']}",
    flush_secs=10,
)

for key, value in list(config.items()):
    if key.endswith("_step"):
        config[key] = eval(value)
    elif key == "optimizer":
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        config[key] = vars(torch.optim)[value](params, **config[f"{key}_config"])
        config["scheduler"] = vars(torch.optim.lr_scheduler)[config["scheduler"]](
            config[key],
            **config["scheduler_config"],
        )

dataframe = pd.read_parquet("hf://datasets/cestwc/law-school-admissions/data/train-00000-of-00001.parquet")
dataframe = dataframe.drop(
    ["enroll", "asian", "black", "hispanic", "white", "missingrace", "urm"],
    axis=1,
)
dataframe = dataframe.replace(to_replace=-1, value=np.nan).dropna(axis=0)

features = dataframe.drop(dataframe.columns[[-1]], axis=1).values
labels = dataframe.iloc[:, -1].values

scaler = StandardScaler()
features[:, 0:6] = scaler.fit_transform(features[:, 0:6])
features[:, 6] = np.where(features[:, 6] == 0, -1, features[:, 6])

dataset = torch.utils.data.TensorDataset(
    torch.tensor(features, dtype=torch.float32),
    torch.tensor(labels, dtype=torch.float32).view(-1, 1),
)
train_size = int(len(features) * 0.7)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(features) - train_size])

train_loader = torch.utils.data.DataLoader(train_set, num_workers=4, batch_size=config["batch_size"])
val_loader = torch.utils.data.DataLoader(val_set, num_workers=4, batch_size=config["batch_size"])

for epoch in range(200):
    if epoch > 0:
        train(model, train_loader=train_loader, epoch=epoch, writer=writer, **config)
    validate(model, val_loader=val_loader, epoch=epoch, writer=writer, **config)

print(model)
outputs = predict(model, predict_binary_classification_step, val_loader=val_loader, **config)
print(outputs.keys(), outputs["predictions"])

writer.flush()
writer.close()
