import streamlit as st
import torch
import torch.nn as nn
from openai import OpenAI
from streamlit_monaco import st_monaco

from utils.utils import (compare_model_flow, compare_model_traces,
                         get_naive_prompt, get_prompt, read_file)

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


def ai_button():
    with st.chat_message("assistant"):
        if st.session_state.is_correct:
            st.session_state.q2_messages.append(
                {"role": "assistant", "content": "You have already solved the task!"})
        else:
            response = st.write_stream(generate_report(task_desc, submission, solution, is_naive=st.session_state.is_naive_prompt))
    st.session_state.q2_messages.append(
        {"role": "assistant", "content": response})


def submit_button(submission: str, solution: str):
    st.session_state.clicked = True
    st.session_state.q2_error_message = ""
    try:
        torch.manual_seed(0)
        exec(submission)
        model = locals()["Model"]()
        torch.manual_seed(0)
        exec(solution)
        expected_model = locals()["ExpectedModel"]()
        st.session_state.is_correct = check_question_2(
            model, expected_model)
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.q2_error_message = e

def check_question_2(model: nn.Module, expected_model: nn.Module):
    # Generate dummy training data
    X_train: torch.Tensor = torch.randint(0, 10000, (64, 100))
    y_train: torch.Tensor = torch.randint(0, 2, (64,))

    # Define loss function
    criterion = nn.NLLLoss()

    num_epochs = 2

    optimizer = torch.optim.Adam(model.parameters())
    losses = train(model, X_train, y_train, optimizer, criterion, num_epochs)

    expected_optimizer = torch.optim.Adam(expected_model.parameters())
    expected_losses = train(expected_model, X_train, y_train, expected_optimizer, criterion, num_epochs)
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
            # Hook tracing
            X_train: torch.Tensor = torch.randint(0, 10000, (64, 100))
            hidden: torch.Tensor = model.init_hidden(X_train.shape[0])
            trace = compare_model_flow(model, expected_model, X_train, hidden)

        # Hooks tracing
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
        ai_button()
        st.rerun()
    for message in reversed(st.session_state.q2_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])