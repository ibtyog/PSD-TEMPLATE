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
    fs = st.slider("Częstotliwość próbkowania (Hz)", 2, 100, 20)
    st.session_state['hz'] = hz
    st.session_state['fs'] = fs
    t = np.linspace(0, 1, 1000)
    t_discrete = np.arange(0, 1, 1/fs)
    x = chirp(t_discrete, f0=hz, f1=hz, t1=1, method='linear', phi=-90)
    real_x = chirp(t, f0=hz, f1=hz, t1=1, method='linear', phi=-90)

    if x is not None and t is not None:
        st.session_state['real_t'] = t
        st.session_state['real_x'] = real_x
        st.session_state['x_discrete'] = x
        st.session_state['t_discrete'] = t_discrete


    fig, ax = plt.subplots()
    ax.plot(t, real_x, label="Sygnał sinusoidalny", alpha=0.5)
    ax.stem(t_discrete, x, linefmt='r-', markerfmt='ro', basefmt=' ', label="Próbki")
    ax.set_title("Sygnał i próbki")
    ax.set_xlabel("Czas (s)")
    ax.set_ylabel("Amplituda")
    ax.legend()
    st.pyplot(fig)

def task_2():
    if 'x_discrete' not in st.session_state:
        st.warning("Najpierw wykonaj Zadanie 1, aby wygenerować sygnał i próbki.")
        return
    t_continuous = st.session_state['real_t']
    x_continuous = st.session_state['real_x']
    t_discrete = st.session_state['t_discrete']
    x_discrete = st.session_state['x_discrete']

    interp_x_values = np.interp(t_continuous, t_discrete, x_discrete)

    error = x_continuous - interp_x_values


    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()    
        ax.plot(t_continuous, x_continuous, label="Sygnał rzeczywisty")
        ax.plot(t_continuous, interp_x_values, label="Interpolacja liniowa", linestyle='--')
        ax.set_title(f'Interpolacja liniowa sinusa, f={st.session_state["hz"]} Hz, fs={st.session_state["fs"]} Hz')
        ax.set_xlabel("Czas (s)")
        ax.set_ylabel("Amplituda")
        ax.legend()
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.plot(t_continuous, error, label="Błąd interpolacji")
        ax.set_title("Błąd interpolacji")
        ax.set_xlabel("Czas (s)")
        ax.set_ylabel("Amplituda")
        ax.legend()
        st.pyplot(fig)

def task_3():
    if 'x_discrete' not in st.session_state:
        st.warning("Najpierw wykonaj Zadanie 1, aby wygenerować sygnał i próbki.")
        return
    
    t_continuous = st.session_state['real_t']
    x_continuous = st.session_state['real_x']
    t_discrete = st.session_state['t_discrete']
    x_discrete = st.session_state['x_discrete']
    fs = st.session_state['fs']
    
    interp_x_sinc = np.zeros(len(t_continuous))
    
    for i in range(len(t_continuous)):
        t = t_continuous[i]
        _sum = 0.0
        
        for n in range(len(t_discrete)):
            t_n = t_discrete[n] 
            x_n = x_discrete[n] 
            
            _sum += x_n * np.sinc((t - t_n) * fs)
            
        interp_x_sinc[i] = _sum

    error_sinc = x_continuous - interp_x_sinc

    c1, c2 = st.columns(2)    
    with c1:
        fig1, ax1 = plt.subplots()    
        ax1.plot(t_continuous, x_continuous, label="Sygnał rzeczywisty", alpha=0.5, linewidth=2)
        ax1.plot(t_continuous, interp_x_sinc, label="Interpolacja sinc", linestyle='--')
        ax1.stem(t_discrete, x_discrete, linefmt='r-', markerfmt='ro', basefmt=' ', label="Próbki")
        ax1.set_title(f'Interpolacja sinc, f={st.session_state["hz"]} Hz, fs={fs} Hz')
        ax1.set_xlabel("Czas (s)")
        ax1.set_ylabel("Amplituda")
        ax1.legend()
        st.pyplot(fig1)
        
    with c2:
        fig2, ax2 = plt.subplots()
        ax2.plot(t_continuous, error_sinc, label="Błąd interpolacji", color="red")
        ax2.set_title("Błąd interpolacji (Whittaker-Shannon)")
        ax2.set_xlabel("Czas (s)")
        ax2.set_ylabel("Błąd")
        ax2.legend()
        st.pyplot(fig2)

tasks_details = {
    1: {
        "title": "Zadanie 1: Pobieranie próbek",
        "description": "Przygotuj w Pythonie kod, który wygeneruje sygnał sinusoidalny o możliwej do zmiany częstotliwości f oraz częstotliwości próbkowania fs. Przygotuj wykres z sygnałem i próbkami pobranymi z zadaną częstotliwością próbkowania fs.",
        "func": task_1,
    },
    2: {
        "title": "Zadanie 2: Interpolacja liniowa",
        "description": "Przygotuj w Pythonie kod, który dokona odcinkami liniowej interpolacji (np. funkcją piecewise dostępną w pakiecie numpy) danych zebranych w zadaniu 1. Wyświetl przebieg błędu interpolacji. ",
        "func": task_2,
    },
    3: {
        "title": "Zadanie 3: Interpolacja Whittakera–Shannona",
        "description": "Przygotuj w Pythonie kod, który dokona interpolacji punktów z zadania 1 z wykorzystaniem równania Whittakera–Shannona: $$x(t) = \\sum_{n=-\\infty}^{\\infty} x_n \\mathrm{sinc}\\left(\\frac{t - nT}{T}\\right)$$ gdzie sinc to funkcja sinus cardinalis (funkcja sinc dostępna jest m.in. w pakiecie scipy). Wyświetl przebieg błędu interpolacji z wykorzystaniem równania Whittakera–Shannona.",
        "func": task_3,
    },
}