from torch.nn import functional as F

from response import attributeResponse
from stochastic import attributeStochastic


def _classification_outputs(net, inputs, labels):
    scores = net(inputs)
    loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="sum")
    correct = ((scores > 0) == labels).sum()
    return scores, loss, correct


def binary_classification_step(net, batch, batch_idx, **kw):
    inputs, labels = batch
    inputs = inputs.to(kw["device"])
    labels = labels.to(kw["device"])

    _, loss, correct = _classification_outputs(net, inputs, labels)
    return {"loss": loss, "correct": correct}


def predict_binary_classification_step(net, batch, batch_idx, **kw):
    inputs, _ = batch
    inputs = inputs.to(kw["device"])
    scores = net(inputs)
    return {"predictions": scores > 0}


def response_step(net, batch, batch_idx, **kw):
    inputs, labels = batch
    # Evaluate each example under both sensitive-feature values.
    inputs = attributeResponse(inputs, index=kw["sensitive_index"], categories=[-1, 1.0])
    labels = labels.repeat(2, 1)
    inputs = inputs.to(kw["device"])
    labels = labels.to(kw["device"])

    _, loss, correct = _classification_outputs(net, inputs, labels)
    loss = loss / 2
    (loss / kw["batch_size"]).backward()
    return {"loss": loss, "correct": correct}


def stochastic_step(net, batch, batch_idx, **kw):
    inputs, labels = batch
    inputs = attributeStochastic(
        inputs,
        index=kw["sensitive_index"],
        epsilon=0.0,
        categories=[-1, 1.0],
    )
    inputs = inputs.to(kw["device"])
    labels = labels.to(kw["device"])

    _, loss, correct = _classification_outputs(net, inputs, labels)
    (loss / kw["batch_size"]).backward()
    return {"loss": loss, "correct": correct}


def erm_step(net, batch, batch_idx, **kw):
    inputs, labels = batch
    inputs = inputs.to(kw["device"])
    labels = labels.to(kw["device"])

    _, loss, correct = _classification_outputs(net, inputs, labels)
    (loss / kw["batch_size"]).backward()
    return {"loss": loss, "correct": correct}


def binary_fair_classification_step(net, batch, batch_idx, **kw):
    inputs, labels = batch
    inputs = inputs.to(kw["device"])
    labels = labels.to(kw["device"])

    scores, loss, correct = _classification_outputs(net, inputs, labels)
    original_predictions = scores > 0

    # Flip the sensitive feature to measure prediction consistency.
    flipped_inputs = inputs.clone()
    flipped_inputs[:, kw["sensitive_index"]] = -flipped_inputs[:, kw["sensitive_index"]]
    flipped_scores, flipped_loss, flipped_correct = _classification_outputs(net, flipped_inputs, labels)
    flipped_predictions = flipped_scores > 0

    consistent = (original_predictions == flipped_predictions).sum()
    return {
        "loss": loss,
        "correct": correct,
        "correct/flip": flipped_correct,
        "loss/flip": flipped_loss,
        "consistent": consistent,
    }
