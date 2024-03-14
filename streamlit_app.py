import streamlit as st
import torch
import torch.nn as nn
from openai import OpenAI
from streamlit_monaco import st_monaco

from utils.utils import check_model, generate_report, read_file

# set basic page config
st.set_page_config(page_title="Selene",
                   page_icon=':books:',
                   layout='wide',
                   initial_sidebar_state='collapsed')

st.title(":books: Selene")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "error_message" not in st.session_state:
    st.session_state.error_message = ""

if 'is_correct' not in st.session_state:
    st.session_state.is_correct = False


def ai_button():
    with st.chat_message("assistant"):
        if st.session_state.is_correct:
            st.session_state.messages.append(
                {"role": "assistant", "content": "You have already solved the task!"})
        else:
            response = st.write_stream(generate_report(client, task_desc, submission, ""))
    st.session_state.messages.append(
        {"role": "assistant", "content": response})


def submit_button(submission: str, solution: str):
    st.session_state.clicked = True
    st.session_state.error_message = ""
    try:
        torch.manual_seed(0)
        exec(submission)
        model = locals()["Model"]()
        torch.manual_seed(0)
        exec(solution)
        expected_model = locals()["SimpleNet1"]()
        st.session_state.is_correct = check_model(
            model, expected_model, nn.CrossEntropyLoss())
    except Exception as e:
        st.session_state.is_correct = False
        st.session_state.error_message = e


left_column, right_column = st.columns(2)
task_desc = read_file("tasks/movie_review_sentiment_classification.md")


with right_column:
    st.subheader("The following code has some issues. Please fix it.")
    with st.container(border=True):
        submission = st_monaco(value=read_file(
            "attempts/movie_review_attempt.py"), height="450px", language="python")
    left, right = st.columns([0.2, 0.8])
    with left:
        solution = read_file("solutions/movie_review_solution.py")
        is_click = st.button("Submit", on_click=submit_button,
                             args=(submission, solution))
    with right:
        if is_click and st.session_state.is_correct:
            st.success("Your solution is correct!")
        elif is_click and not st.session_state.is_correct:
            st.error("Your solution is incorrect. Please try again.")

    if st.session_state.error_message:
        st.exception(st.session_state.error_message)

with left_column:
    st.subheader("Task description")
    with st.container(height=500):
        st.markdown(task_desc)
    if st.button("Help me!"):
        ai_button()
        st.rerun()
    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# if prompt := st.chat_input("What is up?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         stream = client.chat.completions.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         )
#         response = st.write_stream(stream)
#     st.session_state.messages.append(
#         {"role": "assistant", "content": response})
