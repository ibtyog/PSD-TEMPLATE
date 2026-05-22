import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from tasks import tasks_details

st.set_page_config(page_title="PSD - Lista 7", page_icon="🤓", layout="wide")

st.title("PSD - Lista 7")

with st.sidebar:
    st.subheader("Wybór zadania")
    zadanie = st.radio(
        "Wybierz zadanie:",
        (i for i in range(1, len(tasks_details) + 1)),
        format_func=lambda x: f"Zadanie {x}",
    )

if zadanie:
    details = tasks_details[zadanie]
    st.header(details["title"])
    st.info(details["description"])

    results = details["func"]() if "func" in details else ["Brak funkcji do wykonania"]


