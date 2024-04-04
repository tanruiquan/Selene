import streamlit as st
import torch
import torch.nn as nn
from openai import OpenAI
from streamlit_monaco import st_monaco
from torchview import draw_graph

from utils.utils import (LoggingModule, check, compare_layers,
                         get_naive_prompt, get_prompt, read_file)

# set basic page config
st.set_page_config(page_title="Selene",
                   page_icon=':books:',
                   layout='wide',
                   initial_sidebar_state='auto')

st.title(":books: Selene")

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "q3_stream_message_left" not in st.session_state:
    st.session_state.q3_stream_message_left = ""

if "q3_message_left" not in st.session_state:
    st.session_state.q3_message_left = ""

if "q3_stream_message_right" not in st.session_state:
    st.session_state.q3_stream_message_right = ""

if "q3_message_right" not in st.session_state:
    st.session_state.q3_message_right = ""

if "q3_error_message" not in st.session_state:
    st.session_state.q3_error_message = ""

if 'is_correct' not in st.session_state:
    st.session_state.is_correct = False

if "q3_feedback_clicked" not in st.session_state:
    st.session_state.q3_feedback_clicked = False

task_desc = read_file("tasks/movie_review_sentiment_classification.md")
attempt = read_file("attempts/movie_review_attempt2.py")
solution = read_file("solutions/movie_review_solution.py")


def submit_button(submission: str, solution: str) -> None:
    try:
        X_train: torch.Tensor = torch.randn(64, 100)
        y_train: torch.Tensor = torch.randint(0, 1, (64,))
        st.session_state.is_correct = check(
            submission, solution, X_train, y_train)
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.q3_error_message = e


def feedback_button(submission: str, solution: str) -> None:
    st.session_state.q3_feedback_clicked = True
    st.session_state.q3_error_message = ""
    try:
        save_graph(submission, solution)
        report1 = generate_report(
            task_desc, submission, solution, is_naive=False)
        st.session_state.q3_stream_message_left = report1
        report2 = generate_report(
            task_desc, submission, solution, is_naive=True)
        st.session_state.q3_stream_message_right = report2
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.q3_error_message = e


def compare_with_hooks(submission: str, solution: str) -> str:
    X_train: torch.Tensor = torch.randn(64, 100)

    try:
        torch.manual_seed(42)
        exec(submission)
        model = locals()["Model"]()
        with LoggingModule(model) as model:
            logs = model(X_train)

        torch.manual_seed(42)
        exec(solution)
        expected_model = locals()["ExpectedModel"]()
        with LoggingModule(expected_model) as expected_model:
            expected_logs = expected_model(X_train)
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.q3_error_message = e

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
        try:
            torch.manual_seed(0)
            exec(submission)
            model = locals()["Model"]()
            torch.manual_seed(0)
            exec(solution)
            expected_model = locals()["ExpectedModel"]()
            trace = compare_layers(model, expected_model)
        except Exception as e:
            st.session_state.is_correct = False
            st.session_state.q3_error_message = e
            trace = str(e)

        if not trace:
            trace = compare_with_hooks(submission, solution)

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
    st.session_state.q3_error_message = ""
    try:
        torch.manual_seed(42)
        exec(submission)
        model = locals()["Model"]()
        draw_graph(model, input_data=torch.randn(64, 100),
                   graph_name="q3_model", save_graph=True, directory=path)

        torch.manual_seed(42)
        exec(solution)
        expected_model = locals()["ExpectedModel"]()
        draw_graph(expected_model, input_data=torch.randn(64, 100),
                   graph_name="q3_expected_model", save_graph=True, directory=path)
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.q3_error_message = e


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

with left_column:
    st.subheader("Task description")
    with st.container(height=500):
        st.markdown(task_desc)

    q3_feedback_clicked = st.button(
        "Generate feedback", on_click=feedback_button, args=(submission, solution))

with st.empty():
    if st.session_state.q3_error_message:
        st.exception(st.session_state.q3_error_message)


if st.session_state.q3_feedback_clicked:
    graph_feedback, text_feedback = st.tabs(
        ["Graph feedback", "Text feedback"])
    with graph_feedback:
        left_graph, right_graph = st.columns(2)
        with left_graph:
            st.subheader("Your model")
            st.image("graphs/q3_model.gv.png", caption="Your model")
        with right_graph:
            st.subheader("Expected model")
            st.image("graphs/q3_expected_model.gv.png",
                     caption="Expected model")

    with text_feedback:
        left_text, right_text = st.columns(2)
        with left_text:
            if st.session_state.q3_message_left and not q3_feedback_clicked:
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.q3_message_left)
            if st.session_state.q3_stream_message_left and q3_feedback_clicked:
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        st.session_state.q3_stream_message_left)
                st.session_state.q3_message_left = response
        with right_text:
            if st.session_state.q3_message_right and not q3_feedback_clicked:
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.q3_message_right)
            if st.session_state.q3_stream_message_right and q3_feedback_clicked:
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        st.session_state.q3_stream_message_right)
                st.session_state.q3_message_right = response
