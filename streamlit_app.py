import streamlit as st
from code_editor import code_editor
from openai import OpenAI

from utils.utils import read_file

# set basic page config
st.set_page_config(page_title="Selene",
                   page_icon=':books:',
                   layout='wide',
                   initial_sidebar_state='collapsed')

# apply custom css if needed
# with open('assets/styles/style.css') as css:
#     st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


st.title(":books: Selene")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Task description")
    with st.container(height=500):
        st.markdown(read_file(
            "tasks/movie_review_sentiment_classification.md"), unsafe_allow_html=True)

with right_column:
    st.subheader("Enter your code here")
    response_dict = code_editor(
        read_file("attempts/movie_review_attempt.py"), height="500px")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.button("Help me!", on_click=lambda: st.session_state.messages.append(
    {"role": "user", "content": "Help me!"}))

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
