import streamlit as st
from scipy.signal import periodogram, welch, square, sawtooth, chirp, unit_impulse
import numpy as np
from matplotlib import pyplot as plt


def zadanie_1():
    wybor = st.radio(
        "Wybór sygnału",
        [
            "Sinus",
            "Prostokątny",
            "Piłokształtny",
            "Świergotliwy",
            "Superpozycja sin cos",
            "Impuls jednostkowy",
        ],
        horizontal=True,
    )
    st.divider()

    t = np.arange(0, 3, 0.001)
    fs = 1 / 0.001

    signal = None

    if wybor == "Sinus":
        hz = st.slider("Częstotliwość sygnału (Hz)", 1, 50, 1)
        signal = chirp(t, f0=hz, f1=hz, t1=t[-1], method="linear", phi=-90)
    elif wybor == "Prostokątny":
        hz = st.slider("Częstotliwość sygnału (Hz)", 1, 50, 1)
        duty = st.slider("Współczynnik wypełnienia (%)", 0, 100, 50)
        signal = square(2 * np.pi * hz * t, duty=duty / 100)
    elif wybor == "Piłokształtny":
        hz = st.slider("Częstotliwość sygnału (Hz)", 1, 50, 1)
        signal = sawtooth(2 * np.pi * hz * t)
    elif wybor == "Świergotliwy":
        hz = st.slider("Częstotliwość początkowa sygnału (Hz)", 1, 50, 1)
        end_freq = st.slider("Końcowa częstotliwość sygnału (Hz)", 1, 50, 10)
        signal = chirp(t, f0=hz, f1=end_freq, t1=t[-1], method="linear")
    elif wybor == "Superpozycja sin cos":
        hz = st.slider("Częstotliwość sinusa (Hz)", 1, 50, 1)
        cos_freq = st.slider("Częstotliwość cosinusa (Hz)", 1, 50, 1)
        multiply_sin = st.slider("Mnożnik dla sinusa", 0.1, 10.0, 1.0)
        multiply_cos = st.slider("Mnożnik dla cosinusa", 0.1, 10.0, 1.0)
        signal = multiply_sin * np.sin(2 * np.pi * hz * t) + multiply_cos * np.cos(
            2 * np.pi * cos_freq * t
        )
    elif wybor == "Impuls jednostkowy":
        hz = st.slider("Indeks impulsu", 1, len(t) - 1, 1)
        signal = unit_impulse(shape=len(t), idx=hz)

    if signal is not None:

        f_periodogram, Pxx_periodogram = periodogram(signal, fs=fs)

        f_welch, Pxx_welch = welch(signal, fs=fs)

        col = st.columns(3)
        with col[0]:
            fig, ax = plt.subplots()
            ax.plot(t, signal)
            ax.set_title(f"{wybor}, f={hz}Hz")
            ax.set_xlabel("Czas (s)")
            ax.set_ylabel("Amplituda")
            st.pyplot(fig)
        with col[1]:
            fig, ax = plt.subplots()
            ax.semilogy(f_periodogram, Pxx_periodogram)
            ax.set_title("Periodogram")
            ax.set_xlabel("Częstotliwość (Hz)")
            ax.set_ylabel("Gęstość mocy")
            ax.set_xlim(0, 60)
            st.pyplot(fig)
        with col[2]:
            fig, ax = plt.subplots()
            ax.semilogy(f_welch, Pxx_welch)
            ax.set_title("Welch")
            ax.set_xlabel("Częstotliwość (Hz)")
            ax.set_ylabel("Gęstość mocy")
            ax.set_xlim(0, 60)
            st.pyplot(fig)


def zadanie_2():
    wybor = st.radio(
        "Wybór sygnału",
        [
            "Sinus",
            "Prostokątny",
            "Piłokształtny",
            "Świergotliwy",
            "Superpozycja sin cos",
            "Impuls jednostkowy",
        ],
        horizontal=True,
    )
    st.divider()

    t = np.arange(0, 3, 0.001)
    dt = t[1] - t[0]

    signal = None

    if wybor == "Sinus":
        hz = st.slider("Częstotliwość sygnału (Hz)", 1, 50, 1)
        signal = chirp(t, f0=hz, f1=hz, t1=t[-1], method="linear", phi=-90)
    elif wybor == "Prostokątny":
        hz = st.slider("Częstotliwość sygnału (Hz)", 1, 50, 1)
        duty = st.slider("Współczynnik wypełnienia (%)", 0, 100, 50)
        signal = square(2 * np.pi * hz * t, duty=duty / 100)
    elif wybor == "Piłokształtny":
        hz = st.slider("Częstotliwość sygnału (Hz)", 1, 50, 1)
        signal = sawtooth(2 * np.pi * hz * t)
    elif wybor == "Świergotliwy":
        hz = st.slider("Częstotliwość początkowa sygnału (Hz)", 1, 50, 1)
        end_freq = st.slider("Końcowa częstotliwość sygnału (Hz)", 1, 50, 10)
        signal = chirp(t, f0=hz, f1=end_freq, t1=t[-1], method="linear")
    elif wybor == "Superpozycja sin cos":
        hz = st.slider("Częstotliwość sinusa (Hz)", 1, 50, 1)
        cos_freq = st.slider("Częstotliwość cosinusa (Hz)", 1, 50, 1)
        multiply_sin = st.slider("Mnożnik dla sinusa", 0.1, 10.0, 1.0)
        multiply_cos = st.slider("Mnożnik dla cosinusa", 0.1, 10.0, 1.0)
        signal = multiply_sin * np.sin(2 * np.pi * hz * t) + multiply_cos * np.cos(
            2 * np.pi * cos_freq * t
        )
    elif wybor == "Impuls jednostkowy":
        hz = st.slider("Indeks impulsu", 1, len(t) - 1, 1)
        signal = unit_impulse(shape=len(t), idx=hz)

    if signal is not None:
        n = len(signal)
        n_full = 2 * n - 1  

        autocorr = np.correlate(signal, signal, mode='full') / n
        
        psd_wiener = np.abs(np.fft.fft(np.fft.ifftshift(autocorr))) * dt

        freqs = np.fft.fftfreq(n_full, d=dt)

        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        psd_wiener_pos = psd_wiener[pos_mask]
       

        col = st.columns(2)
        with col[0]:
            fig, ax = plt.subplots()
            ax.plot(t, signal)
            ax.set_title(f"{wybor}")
            ax.set_xlabel("Czas (s)")
            ax.set_ylabel("Amplituda")
            st.pyplot(fig)

        with col[1]:
            fig, ax = plt.subplots()


            ax.plot(freqs_pos, psd_wiener_pos)
            ax.set_title("WGM z definicji")
            ax.set_xlabel("Częstotliwość (Hz)")
            ax.set_ylabel("Gęstość mocy")
            ax.set_xlim(0, 60)
            st.pyplot(fig)


def zadanie_3():
    wybor = st.radio(
        "Wybór sygnału",
        [
            "Sinus",
            "Prostokątny",
            "Piłokształtny",
            "Świergotliwy",
            "Superpozycja sin cos",
            "Impuls jednostkowy",
        ],
        horizontal=True,
    )
    st.divider()

    t = np.arange(0, 3, 0.001)
    dt = t[1] - t[0]
    fs = 1 / 0.001

    signal = None

    if wybor == "Sinus":
        hz = st.slider("Częstotliwość sygnału (Hz)", 1, 50, 1)
        signal = chirp(t, f0=hz, f1=hz, t1=t[-1], method="linear", phi=-90)
    elif wybor == "Prostokątny":
        hz = st.slider("Częstotliwość sygnału (Hz)", 1, 50, 1)
        duty = st.slider("Współczynnik wypełnienia (%)", 0, 100, 50)
        signal = square(2 * np.pi * hz * t, duty=duty / 100)
    elif wybor == "Piłokształtny":
        hz = st.slider("Częstotliwość sygnału (Hz)", 1, 50, 1)
        signal = sawtooth(2 * np.pi * hz * t)
    elif wybor == "Świergotliwy":
        hz = st.slider("Częstotliwość początkowa sygnału (Hz)", 1, 50, 1)
        end_freq = st.slider("Końcowa częstotliwość sygnału (Hz)", 1, 50, 10)
        signal = chirp(t, f0=hz, f1=end_freq, t1=t[-1], method="linear")
    elif wybor == "Superpozycja sin cos":
        hz = st.slider("Częstotliwość sinusa (Hz)", 1, 50, 1)
        cos_freq = st.slider("Częstotliwość cosinusa (Hz)", 1, 50, 1)
        multiply_sin = st.slider("Mnożnik dla sinusa", 0.1, 10.0, 1.0)
        multiply_cos = st.slider("Mnożnik dla cosinusa", 0.1, 10.0, 1.0)
        signal = multiply_sin * np.sin(2 * np.pi * hz * t) + multiply_cos * np.cos(
            2 * np.pi * cos_freq * t
        )
    elif wybor == "Impuls jednostkowy":
        hz = st.slider("Indeks impulsu", 1, len(t) - 1, 1)
        signal = unit_impulse(shape=len(t), idx=hz)

    if signal is not None:
        f_periodogram, Pxx_periodogram = periodogram(signal, fs=fs)
        f_welch, Pxx_welch = welch(signal, fs=fs)

        n = len(signal)
        n_full = 2 * n - 1  

        autocorr = np.correlate(signal, signal, mode='full') / n
        
        psd_wiener = np.abs(np.fft.fft(np.fft.ifftshift(autocorr))) * dt

        freqs = np.fft.fftfreq(n_full, d=dt)

        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        psd_wiener_pos = psd_wiener[pos_mask]

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        col = st.columns(2)
        with col[0]:
            fig, ax = plt.subplots()
            ax.plot(t, signal)
            ax.set_title(f"{wybor}")
            ax.set_xlabel("Czas (s)")
            ax.set_ylabel("Amplituda")
            st.pyplot(fig)

        with col[1]:
            fig, ax = plt.subplots()
            ax.plot(freqs_pos, psd_wiener_pos)
            ax.set_title("WGM z definicji")
            ax.set_xlabel("Częstotliwość (Hz)")
            ax.set_ylabel("Gęstość mocy")
            ax.set_xlim(0, 60)
            st.pyplot(fig)

        col2 = st.columns(2)
        with col2[0]:
            fig, ax = plt.subplots()
            ax.plot(f_periodogram, Pxx_periodogram, label="Periodogram")
            ax.set_title("Periodogram")
            ax.set_xlabel("Częstotliwość (Hz)")
            ax.set_ylabel("Moc")
            ax.set_xlim(0, 60)
            ax.legend()
            st.pyplot(fig)

        with col2[1]:
            fig, ax = plt.subplots()
            ax.plot(f_welch, Pxx_welch, label="Metoda Welcha")
            ax.set_title("Metoda Welcha")
            ax.set_xlabel("Częstotliwość (Hz)")
            ax.set_ylabel("Moc")
            ax.set_xlim(0, 60)
            ax.legend()
            st.pyplot(fig)


tasks_details = {
    1: {
        "title": "Zadanie 1: Widmowa gęstość mocy",
        "description": "Dla sygnałów z zadania 1 z listy 1 napisz kod w języku Python, który wyznaczy dla nich widmową gęstość mocy. W tym celu wykorzystaj metody: periodogram i welcha. Metody dostępne są m.in. w bibliotece SciPy. ",
        "func": zadanie_1,
    },
    2: {
        "title": "Zadanie 2: WGM z definicji",
        "description": "Przygotuj w Pythonie kod, który wyznaczy z definicji widmową gęstość mocy. WGM z definicji wyznaczana jest z zależności: $$S_{xx}(f) = \int_{-\infty}^{\infty} R_{xx}(\\tau) e^{-i2\pi f\\tau} d\\tau$$",
        "func": zadanie_2,
    },
    3: {
        "title": "Zadanie 3: Porównanie metod",
        "description": "Dla sygnałów z zadania 1 z listy 1 napisz kod w języku Python, który pozwoli na porównanie wyników uzyskanych dla metod bibliotecznych (periodogram i welch) z samodzielną implementacją z definicji.",
        "func": zadanie_3,
    },
}
