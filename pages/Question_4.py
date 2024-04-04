import streamlit as st
import torch
import torch.nn as nn
from openai import OpenAI
from streamlit_monaco import st_monaco
from torchview import draw_graph

from utils.rnn import generate_report, save_graph, verify
from utils.utils import read_file

# set basic page config
st.set_page_config(page_title="Selene",
                   page_icon=':books:',
                   layout='wide',
                   initial_sidebar_state='auto')

st.title(":books: Selene")

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "q4_stream_message_left" not in st.session_state:
    st.session_state.q4_stream_message_left = ""

if "q4_stream_message_right" not in st.session_state:
    st.session_state.q4_stream_message_right = ""

if "q4_message_left" not in st.session_state:
    st.session_state.q4_message_left = ""

if "q4_message_right" not in st.session_state:
    st.session_state.q4_message_right = ""

if "q4_error_message" not in st.session_state:
    st.session_state.q4_error_message = ""

if "q4_feedback_clicked" not in st.session_state:
    st.session_state.q4_feedback_clicked = False

if 'is_correct' not in st.session_state:
    st.session_state.is_correct = False

task_desc = read_file("tasks/sentiment_analysis.md")
attempt = read_file("attempts/sentiment_analysis_attempt2.py")
solution = read_file("solutions/sentiment_analysis_solution.py")
question = "q4"


def submit_button(submission: str, solution: str):
    st.session_state.q4_error_message = ""
    try:
        st.session_state.is_correct = verify(submission, solution)
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.q4_error_message = e


def feedback_button(submission: str, solution: str) -> None:
    st.session_state.q4_feedback_clicked = True
    st.session_state.q4_error_message = ""
    try:
        save_graph(submission, solution, prefix=question)
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.q4_error_message = e
    report1 = generate_report(
        task_desc, submission, solution, is_naive=False)
    st.session_state.q4_stream_message_left = report1
    report2 = generate_report(
        task_desc, submission, solution, is_naive=True)
    st.session_state.q4_stream_message_right = report2


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


with st.empty():
    if st.session_state.q4_error_message:
        st.exception(st.session_state.q4_error_message)

if st.session_state.q4_feedback_clicked:
    graph_feedback, text_feedback = st.tabs(
        ["Graph Feedback", "Text Feedback"])
    with graph_feedback:
        left_graph, right_graph = st.columns(2)
        with left_graph:
            st.subheader("Your Model")
            st.image(f"graphs/{question}_model.gv.png")
        with right_graph:
            st.subheader("Expected Model")
            st.image(f"graphs/{question}_expected_model.gv.png")

    with text_feedback:
        left_text, right_text = st.columns(2)
        with left_text:
            if st.session_state.q4_message_left and not feedback_clicked:
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.q4_message_left)
            if st.session_state.q4_stream_message_left and feedback_clicked:
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        st.session_state.q4_stream_message_left)
                st.session_state.q4_message_left = response
        with right_text:
            if st.session_state.q4_message_right and not feedback_clicked:
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.q4_message_right)
            if st.session_state.q4_stream_message_right and feedback_clicked:
                with st.chat_message("assistant"):
                    response = st.write_stream(
                        st.session_state.q4_stream_message_right)
                st.session_state.q4_message_right = response
