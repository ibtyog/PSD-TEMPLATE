import streamlit as st
import scipy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def task_1():
    method = st.selectbox(
        "Wybierz metodę", ["Hamming", "Hann", "Blackman", "Dirichlet"]
    )
    signal_option = st.selectbox(
        "Wybierz sygnał", ["Sinus", "Prostokątny", "Piłokształtny", "Losowy"]
    )
    frequency = st.slider("Częstotliwość sygnału (Hz)", 1, 25, 5)
    sampling_rate = st.slider("Częstotliwość próbkowania (Hz)", 1, 200, 200)

    t = np.arange(0, 3, 1 / sampling_rate)
    dt = 1 / sampling_rate

    if signal_option == "Sinus":
        signal = np.sin(2 * np.pi * frequency * t)
    elif signal_option == "Prostokątny":
        signal = scipy.signal.square(2 * np.pi * frequency * t)
    elif signal_option == "Piłokształtny":
        signal = scipy.signal.sawtooth(2 * np.pi * frequency * t)
    elif signal_option == "Losowy":
        signal = np.random.rand(len(t))

    N = len(signal)
    if method == "Hamming":
        window = scipy.signal.windows.hamming(N)
    elif method == "Hann":
        window = scipy.signal.windows.hann(N)
    elif method == "Blackman":
        window = scipy.signal.windows.blackman(N)
    elif method == "Dirichlet":
        window = scipy.signal.windows.boxcar(N)

    windowed_signal = signal * window

    N_fft = 4 * N

    yf = scipy.fft.fft(windowed_signal, n=N_fft)
    xf = scipy.fft.fftfreq(N_fft, d=dt)

    xf_pos = xf[: N_fft // 2]
    yf_abs = np.abs(yf[: N_fft // 2])

    yf_db = 20 * np.log10(yf_abs / np.max(yf_abs) + 1e-10)

    c1, c2, c4, c3 = st.columns(4)

    with c1:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        fig1.suptitle(f"Sygnał: {signal_option}")
        ax1.plot(t, signal)
        ax1.set_xlabel("Czas [s]")
        ax1.set_ylabel("Amplituda")
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(5, 4.78))
        fig2.suptitle(f"Okno: {method}")
        ax2.plot(t, windowed_signal, color="orange")
        ax2.plot(t, window, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Czas [s]")
        st.pyplot(fig2)

    with c4:
        yf_win = scipy.fft.fft(window, n=N_fft)
        
        yf_win_shifted = scipy.fft.fftshift(yf_win)
        
        xf_win = scipy.fft.fftshift(scipy.fft.fftfreq(N_fft))
        
        yf_win_abs = np.abs(yf_win_shifted)
        yf_win_db = 20 * np.log10(yf_win_abs / np.max(yf_win_abs) + 1e-10)

        fig4, ax4 = plt.subplots(figsize=(5, 5.1))
        fig4.suptitle("Widmo okna")
        ax4.plot(xf_win, yf_win_db, color="red")
        ax4.set_xlim(-0.5, 0.5)
        ax4.set_ylim(-100, 5)
        ax4.set_xlabel("Znormalizowana częstotliwość")
        ax4.set_ylabel("Amplituda [dB]")
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)

    with c3:
        fig3, ax3 = plt.subplots(figsize=(5, 5.1))
        fig3.suptitle("Widmo zokienkowanego sygnału")
        ax3.plot(xf_pos, yf_db, color="green")
        ax3.set_xlim(0, sampling_rate / 2)
        ax3.set_ylim(-100, 5)
        ax3.set_xlabel("Częstotliwość [Hz]")
        ax3.set_ylabel("Amplituda [dB]")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)




def task_2():
    method = st.selectbox(
        "Wybierz metodę", ["Hamming", "Hann", "Blackman", "Dirichlet"]
    )
    signal_option = st.selectbox(
        "Wybierz sygnał", ["Sinus", "Prostokątny", "Piłokształtny", "Losowy"]
    )
    frequency_1 = st.slider("Częstotliwość sygnału 1 (Hz)", 1, 25, 5)
    frequency_2 = st.slider("Częstotliwość sygnału 2 (Hz)", 1, 25, 10)
    frequency_3 = st.slider("Częstotliwość sygnału 3 (Hz)", 1, 25, 15)
    sampling_rate = st.slider("Częstotliwość próbkowania (Hz)", 1, 200, 200)

    t = np.arange(0, 3, 1 / sampling_rate)
    dt = 1 / sampling_rate

    signal = None

    if signal_option == "Sinus":
        signal_1 = np.sin(2 * np.pi * frequency_1 * t)
        signal_2 = np.sin(2 * np.pi * frequency_2 * t)
        signal_3 = np.sin(2 * np.pi * frequency_3 * t)
    elif signal_option == "Prostokątny":
        signal_1 = scipy.signal.square(2 * np.pi * frequency_1 * t)
        signal_2 = scipy.signal.square(2 * np.pi * frequency_2 * t)
        signal_3 = scipy.signal.square(2 * np.pi * frequency_3 * t)

    elif signal_option == "Piłokształtny":
        signal_1 = scipy.signal.sawtooth(2 * np.pi * frequency_1 * t)
        signal_2 = scipy.signal.sawtooth(2 * np.pi * frequency_2 * t)
        signal_3 = scipy.signal.sawtooth(2 * np.pi * frequency_3 * t)
    elif signal_option == "Losowy":
        signal_1 = np.random.rand(len(t))
        signal_2 = np.random.rand(len(t))
        signal_3 = np.random.rand(len(t))

    signal = signal_1 + signal_2 + signal_3

    N = len(signal)
    if method == "Hamming":
        window = scipy.signal.windows.hamming(N)
    elif method == "Hann":
        window = scipy.signal.windows.hann(N)
    elif method == "Blackman":
        window = scipy.signal.windows.blackman(N)
    elif method == "Dirichlet":
        window = scipy.signal.windows.boxcar(N)

    windowed_signal = signal * window
    window *= max(signal)

    N_fft = 4 * N

    yf = scipy.fft.fft(windowed_signal, n=N_fft)
    xf = scipy.fft.fftfreq(N_fft, d=dt)

    xf_pos = xf[: N_fft // 2]
    yf_abs = np.abs(yf[: N_fft // 2])

    yf_db = 20 * np.log10(yf_abs / np.max(yf_abs) + 1e-10)

    c1, c2, c3 = st.columns(3)

    with c1:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        fig1.suptitle(f"Sygnał: {signal_option}")
        ax1.plot(t, signal)
        ax1.set_xlabel("Czas [s]")
        ax1.set_ylabel("Amplituda")
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(5, 4.78))
        fig2.suptitle(f"Okno: {method}")
        ax2.plot(t, windowed_signal, color="orange")
        ax2.plot(t, window, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Czas [s]")
        st.pyplot(fig2)

    with c3:
        fig3, ax3 = plt.subplots(figsize=(5, 5.1))
        fig3.suptitle(f"Widmo amplitudowe")
        ax3.plot(xf_pos, yf_db, color="green")
        ax3.set_xlim(0, sampling_rate / 2)
        ax3.set_ylim(-100, 5)
        ax3.set_xlabel("Częstotliwość [Hz]")
        ax3.set_ylabel("Amplituda [dB]")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)


def task_3():
    signal_option = st.selectbox(
        "Wybierz sygnał", ["Sinus", "Prostokątny", "Piłokształtny", "Losowy"]
    )

    frequency = st.slider("Częstotliwość sygnału (Hz)", 1, 25, 5)
    sampling_rate = st.slider("Częstotliwość próbkowania (Hz)", 1, 200, 200)

    t = np.arange(0, 3, 1 / sampling_rate)
    dt = 1 / sampling_rate

    signal = None

    if signal_option == "Sinus":
        signal = np.sin(2 * np.pi * frequency * t)
    elif signal_option == "Prostokątny":
        signal = scipy.signal.square(2 * np.pi * frequency * t)
    elif signal_option == "Piłokształtny":
        signal = scipy.signal.sawtooth(2 * np.pi * frequency * t)
    elif signal_option == "Losowy":
        signal = np.random.rand(len(t)) - 0.5

    N = len(signal)

    yf = scipy.fft.fft(signal)
    xf = scipy.fft.fftfreq(N, d=dt)

    xf_pos = xf[: N // 2]

    yf_abs = 2.0 / N * np.abs(yf[: N // 2])

    c1, c2 = st.columns(2)

    with c1:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        fig1.suptitle(f"Sygnał: {signal_option}")
        ax1.plot(t, signal)
        ax1.set_xlabel("Czas [s]")
        ax1.set_ylabel("Amplituda")
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.suptitle("Widmo amplitudowe")
        ax2.plot(xf_pos, yf_abs, color="green")
        ax2.set_xlim(0, sampling_rate / 2)
        ax2.set_xlabel("Częstotliwość [Hz]")
        ax2.set_ylabel("Amplituda")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)


def task_3():
    signal_option = st.selectbox(
        "Wybierz sygnał", ["Sinus", "Prostokątny", "Piłokształtny", "Losowy"]
    )

    frequency = st.slider("Częstotliwość sygnału (Hz)", 1, 25, 5)
    sampling_rate = st.slider("Częstotliwość próbkowania (Hz)", 1, 200, 200)

    t = np.arange(0, 3, 1 / sampling_rate)
    dt = 1 / sampling_rate

    signal = None

    if signal_option == "Sinus":
        signal = np.sin(2 * np.pi * frequency * t)
    elif signal_option == "Prostokątny":
        signal = scipy.signal.square(2 * np.pi * frequency * t)
    elif signal_option == "Piłokształtny":
        signal = scipy.signal.sawtooth(2 * np.pi * frequency * t)
    elif signal_option == "Losowy":
        signal = np.random.rand(len(t)) - 0.5

    N = len(signal)

    yf = scipy.fft.fft(signal)
    xf = scipy.fft.fftfreq(N, d=dt)

    xf_pos = xf[: N // 2]

    yf_abs = 2.0 / N * np.abs(yf[: N // 2])

    c1, c2 = st.columns(2)

    with c1:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        fig1.suptitle(f"Sygnał: {signal_option}")
        ax1.plot(t, signal)
        ax1.set_xlabel("Czas [s]")
        ax1.set_ylabel("Amplituda")
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.suptitle("Widmo amplitudowe")
        ax2.plot(xf_pos, yf_abs, color="green")
        ax2.set_xlim(0, sampling_rate / 2)
        ax2.set_xlabel("Częstotliwość [Hz]")
        ax2.set_ylabel("Amplituda")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    st.divider()
    st.subheader("Zapisz sygnał do pliku CSV")
    filename = st.text_input("Nazwa pliku", value="moj_sygnal.csv")
    df = pd.DataFrame({"Czas": t, "Amplituda": signal})
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Pobierz sygnał (CSV)", data=csv_data, file_name=filename, mime="text/csv"
    )


def task_4():
    uploaded_file = st.file_uploader("Wczytaj plik z sygnałem (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        t = df.iloc[:, 0].values
        signal = df.iloc[:, 1].values

        dt = t[1] - t[0]
        sampling_rate = 1 / dt

        N = len(signal)

        yf = scipy.fft.fft(signal)
        xf = scipy.fft.fftfreq(N, d=dt)

        xf_pos = xf[: N // 2]
        yf_abs = 2.0 / N * np.abs(yf[: N // 2])

        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            fig1.suptitle("Wczytany sygnał")
            ax1.plot(t, signal)
            ax1.set_xlabel("Czas [s]")
            ax1.set_ylabel("Amplituda")
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            fig2.suptitle("Widmo amplitudowe")
            ax2.plot(xf_pos, yf_abs, color="green")
            ax2.set_xlim(0, sampling_rate / 2)
            ax2.set_xlabel("Częstotliwość [Hz]")
            ax2.set_ylabel("Amplituda")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

    else:
        st.info("Wczytaj plik CSV.")


tasks_details = {
    1: {
        "title": "Zadanie 1: Widmowa gęstość mocy",
        "description": "Przygotuj w Pythonie kod, który przedstawi na wykresie okno Hamminga, Hanna, Blackmana oraz Dirichleta oraz ich widma amplitudowe (pakiet SciPy).",
        "func": task_1,
    },
    2: {
        "title": "Zadanie 2: Okna 3 sygnałów",
        "description": "Przygotuj w Pythonie kod, który wyznaczy widmo sygnału sinusoidalnego o trzech różnych częstotliwościach przy zastosowanych oknach: Hamminga, Hanna, Blackmana oraz Dirichleta",
        "func": task_2,
    },
    3: {
        "title": "Zadanie 3: Widmo z FFT",
        "description": "Przygotuj w Pythonie kod, który wyznaczy widmo sygnału sinusoidalnego z wykorzystaniem Szybkiej Transformaty Fouriera (np. pakiet SciPy oferuje funkcję fft).",
        "func": task_3,
    },
    4: {
        "title": "Zadanie 4: Wczytywanie sygnału z pliku",
        "description": "Przygotuj w Pythonie kod, który wyznaczy widmo dowolnego sygnału załadowanego z pliku np. csv.",
        "func": task_4,
    },
}
