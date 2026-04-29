import torch


def attributeResponse(inputs, index, categories):
    """Duplicate a batch once per sensitive-feature value."""
    categories = torch.as_tensor(categories, dtype=inputs.dtype, device=inputs.device)
    repeated_inputs = inputs.unsqueeze(0).expand(len(categories), *inputs.shape).clone()
    repeated_inputs[torch.arange(len(categories), device=inputs.device), :, index] = categories.unsqueeze(-1)
    return repeated_inputs.view(-1, *inputs.shape[1:])
