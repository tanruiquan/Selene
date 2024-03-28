import streamlit as st
from openai import OpenAI
from streamlit_monaco import st_monaco

from utils.utils import read_file

# set basic page config
st.set_page_config(page_title="Selene",
                   page_icon=':books:',
                   layout='wide',
                   initial_sidebar_state='auto')

st.title(":books: Selene")

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.header("Instruction")
st.markdown("""
- You will receive a task description outlining steps to implement a Neural Network model for a given task.
- You will be provided with a partially implemented model containing errors. 
- Your objective is to correct these errors.
""")

st.header("Example")
task_desc = read_file("tasks/example.md")
attempt = read_file("attempts/example.py")
left_column, right_column = st.columns(2)

with right_column:
    st.subheader("The following code has some issues. Please fix it.")
    with st.container(border=True):
        submission = st_monaco(
            value=attempt, height="450px", language="python")
    left, right = st.columns([0.2, 0.8])
    with left:
        st.button("Submit")

with left_column:
    st.subheader("Task description")
    with st.container(height=500):
        st.markdown(task_desc)

    st.button("Generate feedback")
