import streamlit as st
import torch
import torch.nn as nn
from openai import OpenAI
from streamlit_monaco import st_monaco

from utils.utils import (LoggingModule, compare_model_traces, get_naive_prompt,
                         get_prompt, read_file)

# set basic page config
st.set_page_config(page_title="Selene",
                   page_icon=':books:',
                   layout='wide',
                   initial_sidebar_state='auto')

st.title(":books: Selene")

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "q2_messages" not in st.session_state:
    st.session_state.q2_messages = []

if "q2_error_message" not in st.session_state:
    st.session_state.q2_error_message = ""

if 'is_correct' not in st.session_state:
    st.session_state.is_correct = False

task_desc = read_file("tasks/sentiment_analysis.md")
attempt = read_file("attempts/sentiment_analysis_attempt.py")
solution = read_file("solutions/sentiment_analysis_solution.py")


def submit_button(submission: str, solution: str):
    st.session_state.clicked = True
    st.session_state.q2_error_message = ""
    try:
        st.session_state.is_correct = check_question_2(submission, solution)
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.q2_error_message = e


def check_question_2(submission: str, solution: str) -> bool:
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


def train(model, X_train: torch.Tensor, y_train: torch.Tensor, optimizer, criterion, num_epochs: int = 10, device: str = "cpu"):
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    model.train()

    losses = []
    for epoch in range(num_epochs):
        batch_size, seq_len = X_train.shape
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

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        losses.append(loss.item())
    return losses


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
        if check:
            prev_layer_name, prev_var_name = checked[-1]
            return f"The student made a mistake after calling the {prev_layer_name} layer with the variable name {prev_var_name} and before calling the {curr_layer_name} layer with the variable name {curr_var_name}."
        else:
            return f"The student made a mistake before calling the {curr_layer_name} layer with the variable name {curr_var_name}."
        checked.append(layer)
    return "The student's implementation is correct."


def generate_report(task_desc: str, submission: str, solution: str, is_naive: bool = False):
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
            trace = compare_with_hooks(submission, solution)
            # trace = compare_model_flow(model, expected_model, X_train, hidden)

        prompt = get_prompt(task_desc, submission, solution, trace).strip()

    print(prompt)
    response = st.session_state.client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI teaching assistant helping a student with a coding task. You should answer the student's question in ways that will promote learning and understanding. Do not include a model solution, the corrected code, or automated tests in the response."},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    return response


def save_graph(submission: str, solution: str, path="graphs") -> None:
    st.session_state.q2_error_message = ""
    X_train: torch.Tensor = torch.randint(0, 10000, (64, 100))
    try:
        torch.manual_seed(42)
        exec(submission)
        model = locals()["Model"]()
        draw_graph(model, input_data=X_train, graph_name="q2_model",
                   save_graph=True, directory=path)

        torch.manual_seed(42)
        exec(solution)
        expected_model = locals()["ExpectedModel"]()
        draw_graph(expected_model, input_data=X_train,
                   graph_name="q2_expected_model", save_graph=True, directory=path)
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.q2_error_message = e


left_column, right_column = st.columns(2)


with right_column:
    st.subheader("The following code has some issues. Please fix it.")
    with st.container(border=True):
        submission = st_monaco(
            value=attempt, height="450px", language="python")
    left, right = st.columns([0.2, 0.8])
    with left:
        is_click = st.button("Submit", on_click=submit_button,
                             args=(submission, solution))
    with right:
        if is_click and st.session_state.is_correct:
            st.success("Your solution is correct!")
        elif is_click and not st.session_state.is_correct:
            st.error("Your solution is incorrect. Please try again.")

    if st.session_state.q2_error_message:
        st.exception(st.session_state.q2_error_message)

with left_column:
    st.subheader("Task description")
    with st.container(height=500):
        st.markdown(task_desc)

    if st.button("Generate AI feedback"):
        save_graph(submission, solution)
        if st.session_state.is_correct:
            content = "You have already solved the task!"
            st.chat_message("assistant").write(content)
            st.session_state.q2_messages.append(
                {"role": "assistant", "content": content})
        else:
            stream = generate_report(
                task_desc, submission, solution, is_naive=st.session_state.is_naive_prompt)
            response = st.chat_message("assistant").write_stream(stream)
            st.session_state.q2_messages.append(
                {"role": "assistant", "content": response})
        st.rerun()

    for message in reversed(st.session_state.q2_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
