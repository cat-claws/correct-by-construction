import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from steps import (
    binary_fair_classification_step,
    predict_binary_classification_step,
    response_step,
)
from torchiteration import predict, train_plain as train, validate


config = {
    "dataset": "census",
    "training_step": "response_step",
    "batch_size": 32,
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
    linears=[13] + [50] * 50 + [4],
    num_classes=1,
).to(config["device"])

with torch.no_grad():
    # Break exact symmetry before optimization starts.
    model.layers[1].weight.zero_()
    model.layers[1].weight += np.random.uniform(-1e-10, 1e-10)

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

splits = {
    "train": "data/train-00000-of-00001-78106eba22784d18.parquet",
    "test": "data/test-00000-of-00001-a861e971a12b7550.parquet",
}
data = pd.read_parquet(f"hf://datasets/cestwc/census-income/{splits['train']}")

features = data.drop(data.columns[[2, -1]], axis=1).values
labels = data.iloc[:, -1].values
features[:, 8] = np.where(features[:, 8] == 0, -1, features[:, 8])
features[:, 9] = features[:, 9] / 1000

dataset = torch.utils.data.TensorDataset(
    torch.tensor(features, dtype=torch.float32),
    torch.tensor(labels, dtype=torch.float32).view(-1, 1),
)
train_size = int(len(features) * 0.7)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(features) - train_size])

train_loader = torch.utils.data.DataLoader(train_set, num_workers=4, batch_size=config["batch_size"])
val_loader = torch.utils.data.DataLoader(val_set, num_workers=4, batch_size=config["batch_size"])

for epoch in range(1000):
    if epoch > 0:
        train(model, train_loader=train_loader, epoch=epoch, writer=writer, **config)
    validate(model, val_loader=val_loader, epoch=epoch, writer=writer, **config)

print(model)
outputs = predict(model, predict_binary_classification_step, val_loader=val_loader, **config)
print(outputs.keys(), outputs["predictions"])

writer.flush()
writer.close()
