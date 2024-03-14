from pathlib import Path

import torch
import torch.nn as nn


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


def generate_report(client, task_desc, submission, solution):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": generate_prompt(
                task_desc, submission).strip()},
        ],
        stream=True
    )

    return response 


def generate_prompt(task_desc, submission):
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
