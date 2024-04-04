import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph

from .utils import LoggingModule, compare_layers, get_naive_prompt, get_prompt


def train(model, X_train: torch.Tensor, y_train: torch.Tensor, optimizer, criterion, num_epochs: int = 10, device: str = "cpu") -> list[float]:
    model, X_train, y_train = model.to(
        device), X_train.to(device), y_train.to(device)
    model.train()

    losses = []
    for epoch in range(num_epochs):
        batch_size, _ = X_train.shape
        hidden = model.init_hidden(batch_size)
        if type(hidden) is tuple:
            hidden = (hidden[0].to(device), hidden[1].to(device))  # LSTM
        else:
            hidden = hidden.to(device)  # RNN, GRU

        # Forward pass
        outputs = model(X_train, hidden)
        loss = criterion(outputs, y_train)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


def verify(submission: str, solution: str) -> bool:
    # Generate dummy training data
    X_train: torch.Tensor = torch.randint(0, 10000, (64, 100))
    y_train: torch.Tensor = torch.randint(0, 2, (64,))

    # Define loss function
    criterion = nn.NLLLoss()
    num_epochs = 1

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
    return losses == expected_losses


def save_graph(submission: str, solution: str, prefix: str, path="graphs") -> None:
    X_train: torch.Tensor = torch.randint(0, 10000, (64, 100))

    torch.manual_seed(42)
    exec(submission)
    model = locals()["Model"]()
    hidden = model.init_hidden(X_train.shape[0])
    draw_graph(model, input_data=[X_train, hidden], graph_name=f"{prefix}_model",
               save_graph=True, directory=path)

    torch.manual_seed(42)
    exec(solution)
    expected_model = locals()["ExpectedModel"]()
    hidden = expected_model.init_hidden(X_train.shape[0])
    draw_graph(expected_model, input_data=[X_train, hidden],
               graph_name=f"{prefix}_expected_model", save_graph=True, directory=path)


def compare_with_modules(submission: str, solution: str) -> str:
    torch.manual_seed(42)
    exec(submission)
    model = locals()["Model"]()

    torch.manual_seed(42)
    exec(solution)
    expected_model = locals()["ExpectedModel"]()
    return compare_layers(model, expected_model)


def compare_with_hooks(submission: str, solution: str) -> str:
    X_train: torch.Tensor = torch.randint(0, 10000, (64, 100))

    torch.manual_seed(42)
    exec(submission)
    model = locals()["Model"]()
    hidden = model.init_hidden(X_train.shape[0])
    with LoggingModule(model) as model:
        logs = model(X_train, hidden)

    torch.manual_seed(42)
    exec(solution)
    expected_model = locals()["ExpectedModel"]()
    hidden = expected_model.init_hidden(X_train.shape[0])
    with LoggingModule(expected_model) as expected_model:
        expected_logs = expected_model(X_train, hidden)

    assert len(logs) == len(expected_logs)

    checked = []
    for (layer, output), (expected_layer, expected_output) in zip(logs.items(), expected_logs.items()):
        curr_layer_name, curr_var_name = layer
        if curr_layer_name in ("RNN", "LSTM", "GRU"):
            out, _ = output
            expected_out, _ = expected_output
            if out.permute(1, 0, 2).shape == expected_out.shape:  # Resolve batch_first
                out = out.permute(1, 0, 2)
            if torch.allclose(out, expected_out):
                checked.append(layer)
                continue

            if checked:
                prev_layer_name, prev_var_name = checked[-1]
                return f"The student made a mistake after calling the {prev_layer_name} layer with the variable name {prev_var_name} and before calling the {curr_layer_name} layer with the variable name {curr_var_name}."
            else:
                return f"The student made a mistake before calling the {curr_layer_name} layer with the variable name {curr_var_name}."

        if torch.allclose(output, expected_output):
            checked.append(layer)
            continue
        if checked:
            prev_layer_name, prev_var_name = checked[-1]
            return f"The student made a mistake after calling the {prev_layer_name} layer with the variable name {prev_var_name} and before calling the {curr_layer_name} layer with the variable name {curr_var_name}."
        else:
            return f"The student made a mistake before calling the {curr_layer_name} layer with the variable name {curr_var_name}."
        checked.append(layer)
    return "The student's implementation is correct."


def generate_report(task_desc: str, submission: str, solution: str, is_naive: bool = False):
    if is_naive:
        prompt = get_naive_prompt(task_desc, submission).strip()
        system_prompt = "You are an AI teaching assistant helping a student with a coding task. You should answer the student's question in ways that will promote learning and understanding. Do not include a model solution, the corrected code, or automated tests in the response."
    else:
        try:
            context = compare_with_modules(
                submission, solution) or compare_with_hooks(submission, solution)
        except Exception as e:
            st.session_state.is_correct = False
            st.session_state.q2_error_message = e
            context = str(e)

        prompt = get_prompt(task_desc, submission, solution, context).strip()
        system_prompt = "You are an AI teaching assistant helping a student with a coding task. You should answer the student's question in ways that will promote learning and understanding. Do not include a model solution, the corrected code, or automated tests in the response. Only provide feedback based on the context."

    print(prompt)
    response = st.session_state.client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    return response
