import math

import torch


def stochastic_response(column, epsilon, categories):
    """Apply randomized response to a single categorical feature column."""
    categories = torch.as_tensor(categories, dtype=column.dtype, device=column.device)
    num_categories = len(categories)
    keep_probability = math.exp(epsilon) / (math.exp(epsilon) + num_categories - 1)

    probabilities = torch.full(
        (*column.shape, num_categories),
        (1 - keep_probability) / (num_categories - 1),
        dtype=column.dtype,
        device=column.device,
    )
    correct_indices = column.unsqueeze(-1) == categories.view((1,) * column.dim() + (-1,))
    probabilities[correct_indices] = keep_probability

    sampled_indices = torch.multinomial(probabilities, 1).squeeze(-1)
    return categories[sampled_indices]


def attributeStochastic(inputs, index, epsilon, categories):
    """Clone a batch and perturb one feature with randomized response."""
    perturbed_inputs = inputs.clone()
    perturbed_inputs[:, index] = stochastic_response(inputs[:, index], epsilon, categories)
    return perturbed_inputs
