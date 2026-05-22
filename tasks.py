import streamlit as st
import pywt
from matplotlib import pyplot as plt
import numpy as np
import scipy
import pycwt as wavelet


def task_1():
    wavelet_dict = {
        "Haar": "haar",
        "Daubechies": "db4",
        "Symlets": "sym4",
        "Coiflets": "coif3",
        "Biortogonalna": "bior2.2",
        "Gaussian": "gaus2",
        "Meksykański kapelusz": "mexh",
        "Morleta": "morl"
    }
    
    selected_label = st.selectbox("Wybierz falkę", list(wavelet_dict.keys()))
    wavelet_name = wavelet_dict[selected_label]
    continuous_wavelets = pywt.wavelist(kind='continuous')

    if wavelet_name in continuous_wavelets:
        w = pywt.ContinuousWavelet(wavelet_name)
        psi, x = w.wavefun(level=8)
    else:
        w = pywt.Wavelet(wavelet_name)
        res = w.wavefun(level=8)
        x = res[-1] 
        
        if len(res) == 3:
            psi = res[1]
        elif len(res) == 5:
            psi = res[1] 
    
    c1 = st.container()

    with c1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, np.real(psi))
        ax.set_title(f"Falka: {selected_label}")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Amplituda")
        ax.grid()
        st.pyplot(fig)

def task_2():
    N = st.slider("Rząd falki Daubechies (N)", min_value=1, max_value=20, value=4, step=1)
    b = st.slider("Przesunięcie (b)", min_value=-15.0, max_value=15.0, value=0.0, step=0.5)
    a = st.slider("Skala (a)", min_value=0.2, max_value=5.0, value=1.0, step=0.1)

    wavelet_name = f'db{N}'
    w = pywt.Wavelet(wavelet_name)
    
    res = w.wavefun(level=8)
    psi = res[1]
    x_base = res[2]

    t_transformed = (x_base * a) + b
    
    psi_transformed = psi / np.sqrt(a)

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_base, psi, color='black', alpha=0.2, label="Falka matka (a=1, b=0)")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Amplituda")
        ax.set_title(f"Falka {wavelet_name}")
        st.pyplot(fig)
    
    with c2:
        fig_2, ax_2 = plt.subplots(figsize=(10, 5))
        ax_2.plot(t_transformed, psi_transformed, color='blue', linewidth=2, label=f"$\psi_{{a,b}}(t)$ dla {wavelet_name}")
        ax_2.set_xlim(-20, 20)
        ax_2.set_ylim(-2.0, 2.0)
        ax_2.axhline(0, color='black', linewidth=1, alpha=0.3)
        ax_2.axvline(0, color='black', linewidth=1, alpha=0.3)
        ax_2.set_title(f"Skala: {a} Przesunięcie: {b}")
        ax_2.set_xlabel("Czas")
        ax_2.set_ylabel("Amplituda")
        ax_2.grid(True, alpha=0.3)
        ax_2.legend()
        st.pyplot(fig_2)

def task_3():
    wavelet_dict = {
        "Falka Morleta": wavelet.Morlet(6),
        "Falka Paula": wavelet.Paul(4),
        "Meksykański Kapelusz (DOG)": wavelet.DOG(2)
    }
    
    selected_wavelet_name = st.selectbox("Wybierz falkę analizującą", list(wavelet_dict.keys()))
    


    f0 = st.slider("Częstotliwość początkowa (Hz)", 1, 20, 5, key="pycwt_f0")
    f1 = st.slider("Częstotliwość końcowa (Hz)", 50, 150, 100, key="pycwt_f1")


    dt = 0.001 
    t = np.arange(0, 1, dt)
    signal = scipy.signal.chirp(t, f0=f0, f1=f1, t1=1, method='linear')

    c1 = st.container()

    with c1:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t, signal)
        ax1.set_title("Sygnał świergotliwy (Chirp)")
        ax1.set_xlabel("Czas [s]")
        ax1.set_ylabel("Amplituda")
        st.pyplot(fig1)

    mother = wavelet_dict[selected_wavelet_name] 
    
    s0 = 2 * dt 
    dj = 1 / 12 
    J = int((np.log2(0.5 / dt / s0)) / dj) 

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(signal, dt, dj, s0, J, mother)
    
    power = (np.abs(wave)) ** 2

    c2 = st.container()
    with c2:
        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.pcolormesh(t, freqs, power, shading='gouraud', cmap='viridis')

        ax.set_title(f"{selected_wavelet_name}")
        ax.set_ylabel("Częstotliwość [Hz]")
        ax.set_xlabel("Czas [s]")
        max_y = min(np.max(freqs), max(f0, f1) + 20)
        ax.set_ylim(0, max_y)
        
        fig.colorbar(im, ax=ax, label="Moc")

        st.pyplot(fig)
    

    c3 = st.columns(3)

    for column, (wavelet_name, mother_wavelet) in zip(c3, wavelet_dict.items()):
        wave_c3, _, freqs_c3, _, _, _ = wavelet.cwt(signal, dt, dj, s0, J, mother_wavelet)
        power_c3 = np.abs(wave_c3) ** 2

        with column:
            fig_c3, ax_c3 = plt.subplots(figsize=(5, 4))
            im_c3 = ax_c3.pcolormesh(t, freqs_c3, power_c3, shading='gouraud', cmap='viridis')
            ax_c3.set_title(wavelet_name)
            ax_c3.set_ylabel("Częstotliwość [Hz]")
            ax_c3.set_xlabel("Czas [s]")
            ax_c3.set_ylim(0, max_y)
            fig_c3.colorbar(im_c3, ax=ax_c3, label="Moc")
            st.pyplot(fig_c3)

import streamlit as st
import numpy as np
import scipy.signal
import pywt
import matplotlib.pyplot as plt

def task_4():
    c1, c2 = st.columns(2)
    with c1:
        f0 = st.slider("Częst. początkowa (Hz)", min_value=1, max_value=10, value=2, step=1)
    with c2:
        f1 = st.slider("Częst. końcowa (Hz)", min_value=20, max_value=1000, value=50, step=1)

    t = np.linspace(0, 1, 1000)
    signal = scipy.signal.chirp(t, f0=f0, f1=f1, t1=1, method='linear')

    tab1, tab2, tab3 = st.tabs(["Falka: Haar", "Falka: Daubechies (db4)", "Falka: Symlet (sym8)"])

    wavelets_tabs = [
        (tab1, 'haar', 'Haar (Kanciasta, 1 znikający moment)'),
        (tab2, 'db4', 'Daubechies 4 (Gładka, asymetryczna)'),
        (tab3, 'sym8', 'Symlet 8 (Gładka, niemal symetryczna)')
    ]

    for tab, wav_code, wav_desc in wavelets_tabs:
        with tab:
            coeffs = pywt.wavedec(signal, wav_code, level=1)
            cA1, cD1 = coeffs

            fig, axes = plt.subplots(3, 1, figsize=(10, 6))
            
            axes[0].plot(t, signal, color='black', linewidth=1)
            axes[0].set_title(f"Chirp: {f0} - {f1} Hz)", fontweight='bold')
            axes[0].set_ylabel("Amplituda")
            
            axes[1].plot(cA1, color='#1f77b4', linewidth=1.5) 
            axes[1].set_title("Niskie częstotliwości")
            axes[1].set_ylabel("Współczynniki")
            
            axes[2].plot(cD1, color='#2ca02c', linewidth=1.5) 
            axes[2].set_title("Wysokie częstotliwości")
            axes[2].set_ylabel("Współczynniki")

            for ax in axes:
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_ylim(-2,2)
                x_data = ax.lines[0].get_xdata()
                ax.set_xlim(x_data[0], x_data[-1])

            plt.tight_layout()
            st.pyplot(fig)

tasks_details = {
    1: {
        "title": "Zadanie 1: Falki przeróżne",
        "description": "Przygotuj kod w Pythonie, który wyświetli następujące typy falek: "
        "Haar, Daubechies, Symlets, Coiflets, Biortogonalna,  Gaussian,  Meksykański  kapelusz,  Morleta. "
        "W celu rozwiązania zadania można wykorzystać pakiet pywt.",
        "func": task_1,
    },
    2: {
        "title": "Zadanie 2: Falka Daubechies",
        "description": "Przygotuj kod w Pythonie, który wyświetli falkę Daubechies w różnych wersjach "
        "(db1, db2, itd.) i dla różnych parametrów.",
        "func": task_2,
    },
    3: {
        "title": "Zadanie 3: Dekompozycja sygnału świergotliwego (Chirp)",
        "description": "Przygotuj  kod  w  Pythonie, który dokona dekompozycji sygnału świergotliwego "
        "(chirp signal) z wykorzystaniem trzech różnych falek. Uzyskane wyniki wyświetl w czytelnej postaci.",
        "func": task_4,
    },
}
