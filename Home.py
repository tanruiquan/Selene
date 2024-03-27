import streamlit as st
from openai import OpenAI

# set basic page config
st.set_page_config(page_title="Selene",
                   page_icon=':books:',
                   layout='wide',
                   initial_sidebar_state='auto')

st.title(":books: Selene")

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "is_naive_prompt" not in st.session_state:
    st.session_state.is_naive_prompt = True

st.header("Navigation")
st.page_link("Home.py", label="Home", icon="🏠")
st.page_link("pages/Question_1.py", icon="1️⃣")
st.page_link("pages/Question_2.py", icon="2️⃣")

st.header("Instruction")
st.markdown("""
- You will receive a task description outlining steps to implement a Neural Network model for a given task.
- You will be provided with a partially implemented model containing errors. 
- Your objective is to correct these errors.
""")
