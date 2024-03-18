from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def read_file(file_path):
    return Path(file_path).read_text()


def check_model(model: nn.Module, expected_model: nn.Module, criterion: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer = torch.optim.Adam, num_epochs: int = 10, device: str = "cpu"):
    # Question 1
    X_train: torch.Tensor = torch.randn(64, 100)
    y_train: torch.Tensor = torch.randint(0, 1, (64,))
    losses = train(model, X_train, y_train, criterion, optimizer(
        model.parameters()), num_epochs, device)
    expected_losses = train(expected_model, X_train, y_train, criterion, optimizer(
        expected_model.parameters()), num_epochs, device)
    return losses == expected_losses


def train(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, criterion: torch.nn.modules.loss._Loss, optimizer: torch.optim.Optimizer = torch.optim.Adam, num_epochs: int = 10, device: str = "cpu"):
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    model.train()

    losses = []
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        losses.append(loss.item())
    return losses


def generate_report(client, task_desc: str, submission: str, solution: str, is_naive: bool = False):
    if is_naive:
        prompt = get_naive_prompt(task_desc, submission).strip()
    else:
        # Modules tracing
        torch.manual_seed(0)
        exec(submission)
        model = locals()["Model"]()
        torch.manual_seed(0)
        exec(solution)
        expected_model = locals()["ExpectedModel"]()
        trace = compare_model_traces(model, expected_model)

        if not trace:
            # Hook tracing
            trace = "Hook"
            raise NotImplementedError()

        # Hooks tracing
        prompt = get_prompt(task_desc, submission, solution, trace).strip()

    print(prompt)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI teaching assistant helping a student with a coding task. You should answer the student's question in ways that will promote learning and understanding. Do not include a model solution, the corrected code, or automated tests in the response."},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    return response 


def get_naive_prompt(task_desc: str, submission: str) -> str:
    return f"""
# Task description
{task_desc}

# My Code
{submission}

# Question
I am learning Natural Language Processing where we use the PyTorch
library. I have been given an NLP task with the above task
description. I have written code given above. Please provide suggestions on how 
I could fix my code so that it fulfils the requirements in 
the task description. Do not include a model solution, the corrected code,
or automated tests in the response.

# Answer
"""

def compare_model_traces(submitted_model: nn.Module, expected_model: nn.Module) -> str:
    result = []

    submitted_modules = get_modules(submitted_model)
    expected_modules = get_modules(expected_model)
    print(f"Submitted: {submitted_modules}, \nExpected: {expected_modules}")

    if len(submitted_modules) > len(expected_modules):
        c1 = Counter(submitted_modules)
        c2 = Counter(expected_modules)
        diff = c1 - c2
        return f"You have the following extra layers: {diff.items()}"
    elif len(expected_modules) > len(submitted_modules):
        c1 = Counter(map(lambda x: type(x), expected_modules))
        c2 = Counter(map(lambda x: type(x), submitted_modules))
        diff = c1 - c2
        missing = ", ".join(map(lambda x: x.__name__, diff.keys()))
        return f"I am missing the following layers: {missing}. I could not really figure where it is missing. Could you help me with that?"

    module_count = 1
    linear_layer_count = 1
    activation_function_count = 1
    dropout_count = 1
    for s, e in zip(submitted_modules, expected_modules):
        if type(s) != type(e):
            result.append(
                f"Your {make_ordinal(module_count)} module is of type {type(s)} while the expected module is of type {type(e)}")
        elif isinstance(s, nn.Linear):
            if s.in_features != e.in_features or s.out_features != e.out_features:
                result.append(f"Your {make_ordinal(linear_layer_count)} linear layer have an input size of {s.in_features} and an output size of {s.out_features} while the expected linear layer have an input size of {e.in_features} and an output size of {e.out_features}")
            if s.bias is None and e.bias is not None:
                result.append(
                    f"Your {make_ordinal(linear_layer_count)} linear layer is missing a bias term")
            if s.bias is not None and e.bias is None:
                result.apennd(
                    f"Your {make_ordinal(linear_layer_count)} linear layer is not suppose to have a bias term")
            linear_layer_count += 1
        elif is_activation_function(s) and type(s) != type(e):
            result.append(
                f"Your {make_ordinal(activation_function_count)} activation function is of type {type(s)} while the expected activation function is of type {type(e)}")
            activation_function_count += 1
        elif isinstance(s, nn.Dropout) and s.p != e.p:
            result.append(
                f"Your {make_ordinal(dropout_count)} dropout probability is {s.p} while the expected dropout probability is {e.p}")
            dropout_count += 1
        module_count += 1
    return ".".join(result)

def get_modules(model: nn.Module) -> list[nn.Module]:
    """
    Retrieves a list of all modules used in the given PyTorch model.

    The function also converts the functional calls to their respective modules
    as a side effect.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        list[nn.Module]: A list of all modules used in the model.
    """
    traced = torch.fx.symbolic_trace(model)
    ref_modules = dict(traced.named_modules())

    modules = []
    for n in traced.graph.nodes:
        if n.op == "call_module":
            modules.append(ref_modules[n.target])
        elif n.target == "sigmoid":
            modules.append(nn.Sigmoid())
        elif n.target == "tanh":
            modules.append(nn.Tanh())
        elif n.op == "call_function":
            if n.target == F.relu:
                modules.append(nn.ReLU())
            elif n.target == F.sigmoid or n.target == torch.sigmoid:
                modules.append(nn.Sigmoid())
            elif n.target == F.tanh or n.target == torch.tanh:
                modules.append(nn.Tanh())
            elif n.target == F.softmax:
                dim = n.kwargs.get("dim")
                modules.append(nn.Softmax(dim))
            elif n.target == F.log_softmax:
                dim = n.kwargs.get("dim")
                modules.append(nn.LogSoftmax(dim))
            elif n.target == F.dropout:
                p = n.kwargs.get("p")
                inplace = n.kwargs.get("inplace")
                modules.append(nn.Dropout(p, inplace))
            elif n.target == F.dropout1d:
                p = n.kwargs.get("p")
                inplace = n.kwargs.get("inplace")
                modules.append(nn.Dropout1d(p, inplace))
            elif n.target == F.dropout2d:
                p = n.kwargs.get("p")
                inplace = n.kwargs.get("inplace")
                modules.append(nn.Dropout2d(p, inplace))
            elif n.target == F.max_pool1d:
                _, kernal_size = n.args
                modules.append(nn.MaxPool1d(kernal_size, **n.kwargs))
            elif n.target == F.max_pool2d:
                _, kernal_size = n.args
                modules.append(nn.MaxPool2d(kernal_size, **n.kwargs))
    return modules


def make_ordinal(n: int) -> str:
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

def is_activation_function(module: nn.Module) -> bool:
    all_activation_functions = (nn.ReLU, nn.RReLU, nn.Hardtanh, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid, nn.Tanh, nn.SiLU, nn.Mish, nn.Hardswish, nn.ELU, nn.CELU, nn.SELU, nn.GLU, nn.GELU, nn.Hardshrink, nn.LeakyReLU, nn.LogSigmoid, nn.Softplus, nn.Softshrink, nn.MultiheadAttention, nn.PReLU, nn.Softsign, nn.Tanhshrink, nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax)
    return isinstance(module, all_activation_functions)



def get_prompt(task_desc: str, submission: str, solution: str, trace: str) -> str:
    return f"""
# Task description
{task_desc}

# My Code
{submission}

# Question
{trace}

# Answer
"""
