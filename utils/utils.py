from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def read_file(file_path):
    return Path(file_path).read_text()


def check(submission: str, solution: str, X_train: torch.Tensor, y_train: torch.Tensor) -> bool:
    criterion = nn.NLLLoss()
    num_epochs = 5

    torch.manual_seed(42)
    exec(submission)
    model = locals()["Model"]()
    optimizer = torch.optim.Adam(model.parameters())
    losses = train(model, X_train, y_train, optimizer, criterion, num_epochs)

    torch.manual_seed(42)
    exec(solution)
    expected_model = locals()["ExpectedModel"]()
    expected_optimizer = torch.optim.Adam(expected_model.parameters())
    expected_losses = train(expected_model, X_train,
                            y_train, expected_optimizer, criterion, num_epochs)
    return losses == expected_lossess


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


def get_naive_prompt(task_desc: str, submission: str) -> str:
    return f"""
# Task description
{task_desc}

# My Code
{submission}

# Answer
"""


def compare_model_traces(submitted_model: nn.Module, expected_model: nn.Module) -> str:
    result = []

    submitted_modules = get_modules(submitted_model)
    expected_modules = get_modules(expected_model)
    print(f"Submitted: {submitted_modules}, \nExpected: {expected_modules}")
    c1 = Counter(map(lambda x: type(x), expected_modules))
    c2 = Counter(map(lambda x: type(x), submitted_modules))
    if len(submitted_modules) > len(expected_modules):
        diff = c2 - c1
        extra = ", ".join(map(lambda x: x.__name__, diff.keys()))
        return f"The student has the following extra layers: {extra}."
    elif len(expected_modules) > len(submitted_modules):
        diff = c1 - c2
        print(diff)
        missing = ", ".join(map(lambda x: x.__name__, diff.keys()))
        return f"The student has the following missing layers: {missing}."

    module_count = 1
    linear_layer_count = 1
    activation_function_count = 1
    dropout_count = 1
    for s, e in zip(submitted_modules, expected_modules):
        if type(s) != type(e):
            result.append(
                f"The student's {make_ordinal(module_count)} layer is a {s.__class__.__name__} layer while the expected layer is a {e.__class__.__name__} layer.")
        elif isinstance(s, nn.Linear):
            if s.in_features != e.in_features or s.out_features != e.out_features:
                result.append(
                    f"The student's {make_ordinal(linear_layer_count)} linear layer have an input size of {s.in_features} and an output size of {s.out_features} while the expected linear layer have an input size of {e.in_features} and an output size of {e.out_features}.")
            if s.bias is None and e.bias is not None:
                result.append(
                    f"The student's {make_ordinal(linear_layer_count)} linear layer is missing a bias term.")
            if s.bias is not None and e.bias is None:
                result.apennd(
                    f"The student's {make_ordinal(linear_layer_count)} linear layer is not suppose to have a bias term.")
            linear_layer_count += 1
        elif is_activation_function(s) and type(s) != type(e):
            result.append(
                f"The student's {make_ordinal(activation_function_count)} activation function is a {s.__class__.__name__} function while the expected activation function is a {e.__class__.__name__}.")
            activation_function_count += 1
        elif isinstance(s, nn.Dropout) and s.p != e.p:
            result.append(
                f"The student's {make_ordinal(dropout_count)} dropout layer has a dropout probability of {s.p} while the expected dropout probability is {e.p}.")
            dropout_count += 1
        module_count += 1
    return " ".join(result)


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
    all_activation_functions = (nn.ReLU, nn.RReLU, nn.Hardtanh, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid, nn.Tanh, nn.SiLU, nn.Mish, nn.Hardswish, nn.ELU, nn.CELU, nn.SELU, nn.GLU, nn.GELU,
                                nn.Hardshrink, nn.LeakyReLU, nn.LogSigmoid, nn.Softplus, nn.Softshrink, nn.MultiheadAttention, nn.PReLU, nn.Softsign, nn.Tanhshrink, nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax)
    return isinstance(module, all_activation_functions)


def get_prompt(task_desc: str, submission: str, solution: str, trace: str) -> str:
    return f"""
# Task description
{task_desc}

# My Code
{submission}

# Context
{trace}

# Answer
"""


class LoggingModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.log = {}

    def __enter__(self):
        self.handles = []
        for name, layer in self.model.named_modules():
            layer.name = name
            handle = layer.register_forward_hook(self.hook)
            self.handles.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def forward(self, *x):
        _ = self.model(*x)
        return self.log

    def hook(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        self.log[(module.__class__.__name__, module.name)] = output


def compare_model_flow(submitted_model: nn.Module, expected_model: nn.Module, *input: torch.Tensor) -> str:
    # Wrap the modules in a LoggingModule
    with LoggingModule(submitted_model) as submitted, LoggingModule(expected_model) as expected:
        submitted_logs = submitted(*input)
        expected_logs = expected(*input)

    assert len(submitted_logs) == len(
        expected_logs), "The number of layers in the submitted model does not match the expected model"
    print(f"Submitted logs: {submitted_logs.keys()}")
    print(f"Exptected logs: {expected_logs.keys()}")
    submitted_checked = []
    expected_checked = []
    for (s_layer, s_output), (e_layer, e_output) in zip(submitted_logs.items(), expected_logs.items()):
        curr_class_name, curr_var_name = s_layer
        if s_layer[0] in ("RNN", "LSTM", "GRU"):
            s_out = s_output[0]
            e_out = e_output[0]
            # resolve the batch_first issue
            if s_out.shape[0] == e_out.shape[1] and s_out.shape[1] == e_out.shape[0]:
                s_out = s_out.transpose(0, 1)
            # print(f"Shape of s_out: {s_out.shape}, Shape of e_out: {e_out.shape}")
            if not torch.allclose(s_out, e_out) and len(submitted_checked) == 0:
                return f"The student made a mistake before calling the {curr_class_name} layer with the variable name {curr_var_name}."
            if not torch.allclose(s_out, e_out) and len(submitted_checked) > 0:
                prev_class_name, prev_var_name = submitted_checked[-1]
                return f"The student made a mistake after calling the {prev_class_name} layer with the variable name {prev_var_name} and before calling the {curr_class_name} layer with the variable name {curr_var_name}."
        elif not torch.allclose(s_output, e_output) and len(submitted_checked) == 0:
            return f"The student made a mistake before calling the {curr_class_name} layer with the variable name {curr_var_name}."
        elif not torch.allclose(s_output, e_output) and len(submitted_checked) > 0:
            prev_class_name, prev_var_name = submitted_checked[-1]
            return f"The student made a mistake after calling the {prev_class_name} layer with the variable name {prev_var_name} and before calling the {curr_class_name} layer with the variable name {curr_var_name}."
        submitted_checked.append(s_layer)
        expected_checked.append(e_layer)
    return f"The student likely made a mistake after calling the {curr_class_name} layer with the variable name {curr_var_name}."
