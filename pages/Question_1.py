import streamlit as st
import torch
import torch.nn as nn
from openai import OpenAI
from streamlit_monaco import st_monaco
from torchview import draw_graph

from utils.utils import (LoggingModule, check_model, compare_model_traces,
                         get_naive_prompt, get_prompt, read_file)

# set basic page config
st.set_page_config(page_title="Selene",
                   page_icon=':books:',
                   layout='wide',
                   initial_sidebar_state='auto')

st.title(":books: Selene")

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "stream_message_left" not in st.session_state:
    st.session_state.stream_message_left = ""

if "message_left" not in st.session_state:
    st.session_state.message_left = ""

if "stream_message_right" not in st.session_state:
    st.session_state.stream_message_right = ""

if "message_right" not in st.session_state:
    st.session_state.message_right = ""

if "error_message" not in st.session_state:
    st.session_state.error_message = ""

if 'is_correct' not in st.session_state:
    st.session_state.is_correct = False

if "feedback_clicked" not in st.session_state:
    st.session_state.feedback_clicked = False

task_desc = read_file("tasks/movie_review_sentiment_classification.md")
attempt = read_file("attempts/movie_review_attempt.py")
solution = read_file("solutions/movie_review_solution.py")


def submit_button(submission: str, solution: str) -> None:
    st.session_state.clicked = True
    st.session_state.error_message = ""
    try:
        torch.manual_seed(0)
        exec(submission)
        model = locals()["Model"]()
        torch.manual_seed(0)
        exec(solution)
        expected_model = locals()["ExpectedModel"]()
        st.session_state.is_correct = check_model(
            model, expected_model, nn.CrossEntropyLoss())
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.error_message = e


def feedback_button(submission: str, solution: str) -> None:
    st.session_state.feedback_clicked = True
    save_graph(submission, solution)
    report1 = generate_report(task_desc, submission, solution, is_naive=True)
    st.session_state.stream_message_left = report1
    report2 = generate_report(task_desc, submission, solution, is_naive=False)
    st.session_state.stream_message_right = report2


def compare_with_hooks(submission: str, solution: str) -> str:
    X_train: torch.Tensor = torch.randn(64, 100)

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

        prompt = get_prompt(task_desc, submission, solution, trace).strip()

    print(prompt)
    response = st.session_state.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI teaching assistant helping a student with a coding task. You should answer the student's question in ways that will promote learning and understanding. Do not include a model solution, the corrected code, or automated tests in the response."},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    return response


def save_graph(submission: str, solution: str, path="graphs") -> None:
    st.session_state.error_message = ""
    try:
        torch.manual_seed(42)
        exec(submission)
        model = locals()["Model"]()
        draw_graph(model, input_data=torch.randn(64, 100),
                   graph_name="model", save_graph=True, directory=path)

        torch.manual_seed(42)
        exec(solution)
        expected_model = locals()["ExpectedModel"]()
        draw_graph(expected_model, input_data=torch.randn(64, 100),
                   graph_name="expected_model", save_graph=True, directory=path)
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.error_message = e


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

    feedback_clicked = st.button(
        "Generate feedback", on_click=feedback_button, args=(submission, solution))
    # if feedback_clicked:
    #     save_graph(submission, solution)

    # if st.button("Generate AI feedback"):
    #     save_graph(submission, solution)
    #     if st.session_state.is_correct:
    #         content = "You have already solved the task!"
    #         st.chat_message("assistant").write(content)
    #         st.session_state.messages.append(
    #             {"role": "assistant", "content": content})
    #     else:
    #         chat_message = st.chat_message("assistant")
    #         stream = generate_report(
    #             task_desc, submission, solution, is_naive=st.session_state.is_naive_prompt)
    #         response = st.chat_message("assistant").write_stream(stream)
    #         st.session_state.messages.append(
    #             {"role": "assistant", "content": response})
    #     st.rerun()

    # for message in reversed(st.session_state.messages):
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

with st.empty():
    if st.session_state.error_message:
        st.exception(st.session_state.error_message)


if st.session_state.feedback_clicked:
    graph_feedback, text_feedback = st.tabs(
        ["Graph feedback", "Text feedback"])
    with graph_feedback:
        left_graph, right_graph = st.columns(2)
        with left_graph:
            st.subheader("Your model")
            st.image("graphs/model.gv.png", caption="Your model")
        with right_graph:
            st.subheader("Expected model")
            st.image("graphs/expected_model.gv.png", caption="Expected model")

    with text_feedback:
        left_text, right_text = st.columns(2)
        with left_text:
            print(
                f"message_left: {st.session_state.message_left} and feedback_clicked: {feedback_clicked}")
            if st.session_state.message_left and not feedback_clicked:
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.message_left)
            if st.session_state.stream_message_left and feedback_clicked:
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        st.session_state.stream_message_left)
                st.session_state.message_left = response
        with right_text:
            if st.session_state.message_right and not feedback_clicked:
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.message_right)
            if st.session_state.stream_message_right and feedback_clicked:
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        st.session_state.stream_message_right)
                st.session_state.message_right = response

# left_feedback, right_feedback = st.columns(2)

# with left_feedback:
#     if st.session_state.feedback_clicked:
#         st.subheader("Your model")
#         st.image("graphs/model.gv.png", caption="Your model")

#         if st.session_state.stream_message_left:
#             assistant = st.chat_message("assistant")
#         if st.session_state.message_left:
#             with st.chat_message("assistant"):
#                 st.markdown(st.session_state.message_left[-1]['content'])
#         # for msg in reversed(st.session_state.message_left):
#         #     with st.chat_message("assistant"):
#         #         st.markdown(msg['content'])
#         # with st.chat_message("assistant"):
#         #     stream = generate_report(task_desc, submission, solution, is_naive=True)
#         #     response = st.write_stream(stream)

# with right_feedback:
#     if st.session_state.feedback_clicked:
#         st.subheader("Expected model")
#         st.image("graphs/expected_model.gv.png", caption="Expected model")

#         if st.session_state.stream_message_left:
#             response = assistant.write_stream(
#                 st.session_state.stream_message_left)
#             st.session_state.message_left.append(
#                 {"role": "assistant", "content": response})
#             st.session_state.stream_message_left = ""

#         if st.session_state.message_right:
#             with st.chat_message("assistant"):
#                 st.markdown(st.session_state.message_right[-1]['content'])
#         if st.session_state.stream_message_right:
#             with st.chat_message("assistant"):
#                 response = st.write_stream(st.session_state.stream_message_right)
#                 st.session_state.message_right.append(
#                     {"role": "assistant", "content": response})
#                 st.session_state.stream_message_right = ""
        # for msg in reversed(st.session_state.message_right):
        #     with st.chat_message("assistant"):
        #         st.markdown(msg['content'])
