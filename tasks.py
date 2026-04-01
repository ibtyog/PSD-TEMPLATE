import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.interpolate import CubicSpline

def random_code():
    x = st.slider("Liczba wyników do wygenerowania", 1, 10, 5)
    return (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

st.session_state['t'] = None
st.session_state['x'] = None

def task_1():
    hz = st.slider("Częstotliwość sygnału (Hz)", 1, 10, 5)
    fs = st.slider("Częstotliwość próbkowania (Hz)", 2, 500, 500)
    st.session_state['hz'] = hz
    st.session_state['fs'] = fs
    t = np.linspace(0, 1, fs, endpoint=False)
    x = chirp(t, f0=hz, f1=hz, t1=1, method='linear', phi=-90)

    if x is not None and t is not None:
        st.session_state['t'] = t
        st.session_state['x'] = x
    fig, ax = plt.subplots()
    ax.plot(t, x, label="Sygnał sinusoidalny")
    plt.stem(t, x, linefmt='r-', markerfmt='ro', basefmt=' ', label="Próbki")
    ax.set_title("Sygnał i próbki")
    ax.set_xlabel("Czas (s)")
    ax.set_ylabel("Amplituda")
    ax.legend()
    st.pyplot(fig)

def task_2():
    if st.session_state['t'] is not None and st.session_state['x'] is not None:
        interp_x = CubicSpline(st.session_state['t'], st.session_state['x'])
    else:
        st.warning("Najpierw wykonaj Zadanie 1, aby wygenerować sygnał i próbki.")

    real_t = np.linspace(0, 1, 2000)
    real_x = chirp(real_t, f0=st.session_state['hz'], f1=st.session_state['hz'], t1=1, method='linear', phi=-90)
    interp_x_values = interp_x(real_t)

    fig, ax = plt.subplots()
    ax.plot(real_t, real_x, label="Sygnał rzeczywisty")
    ax.plot(real_t, interp_x_values, label="Interpolacja liniowa", linestyle='--')
    ax.set_title("Interpolacja liniowa vs sygnał rzeczywisty")
    ax.set_xlabel("Czas (s)")
    ax.set_ylabel("Amplituda")
    ax.legend()
    st.pyplot(fig)


tasks_details = {
    1: {
        "title": "Zadanie 1: Widmowa gęstość mocy",
        "description": "Przygotuj w Pythonie kod, który wygeneruje sygnał sinusoidalny o możliwej do zmiany częstotliwości f oraz częstotliwości próbkowania fs. Przygotuj wykres z sygnałem i próbkami pobranymi z zadaną częstotliwością próbkowania fs.",
        "func": task_1,
    },
    2: {
        "title": "Zadanie 2: WGM z definicji",
        "description": "Przygotuj w Pythonie kod, który dokona odcinkami liniowej interpolacji (np. funkcją piecewise dostępną w pakiecie numpy) danych zebranych w zadaniu 1. Wyświetl przebieg błędu interpolacji. ",
        "func": task_2,
    },
    3: {
        "title": "Zadanie 3: Porównanie metod",
        "description": "Przygotuj w Pythonie kod, który dokona interpolacji punktów z zadania 1 z wykorzystaniem równania Whittakera–Shannona: $$x(t) = \\sum_{n=-\\infty}^{\\infty} x_n \\mathrm{sinc}\\left(\\frac{t - nT}{T}\\right)$$ gdzie sinc to funkcja sinus cardinalis (funkcja sinc dostępna jest m.in. w pakiecie scipy). Wyświetl przebieg błędu interpolacji z wykorzystaniem równania Whittakera–Shannona.",
    },
}