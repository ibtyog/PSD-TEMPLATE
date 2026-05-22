import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import emd
import scipy.fft as fft
import pandas as pd


def random_code():
    x = st.slider("Liczba wyników do wygenerowania", 1, 10, 5)
    return (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def task_1():
    fs = 1000
    t_max = 3.0
    t = np.linspace(0, t_max, int(fs * t_max), endpoint=False)

    signal_type = st.selectbox(
        "Wybierz sygnał do analizy",
        [
            "Sinus",
            "Prostokątny",
            "Piłokształtny",
            "Świergotliwy (Chirp)",
            "Superpozycja (sin + cos)",
            "Impuls jednostkowy",
        ],
    )
    hz = st.slider("Częstotliwość sygnału głównego (Hz)", 1, 100, 10)

    if signal_type == "Superpozycja (sin + cos)":
        hz2 = st.slider("Częstotliwość drugiego sygnału (Hz)", 1, 100, 20)

    if signal_type == "Sinus":
        y = np.sin(2 * np.pi * hz * t)
    elif signal_type == "Prostokątny":
        y = sig.square(2 * np.pi * hz * t)
    elif signal_type == "Piłokształtny":
        y = sig.sawtooth(2 * np.pi * hz * t)
    elif signal_type == "Świergotliwy (Chirp)":
        hz3 = st.slider("Częstotliwość końcowa (Hz)", 1, 100, 50)
        y = sig.chirp(t, f0=hz, f1=hz3, t1=t_max, method="linear")

    elif signal_type == "Superpozycja (sin + cos)":
        y = np.sin(2 * np.pi * hz * t) + np.cos(2 * np.pi * hz2 * t)

    elif signal_type == "Impuls jednostkowy":
        y = np.zeros_like(t)
        sample_idx = st.slider(
            "Indeks próbki impulsu (0 do {})".format(len(t) - 1),
            0,
            len(t) - 1,
            len(t) // 2,
        )
        y[sample_idx] = 1.0

    fig1, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(t, y, color="black", linewidth=1)
    ax1.set_title(f"Sygnał w dziedzinie czasu: {signal_type}")
    ax1.set_xlabel("Czas [s]")
    ax1.set_ylabel("Amplituda")
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    c1 = st.container()
    with c1:
        st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    f_spec, t_spec, Sxx = sig.spectrogram(y, fs, nperseg=256, noverlap=128)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    im = ax2.pcolormesh(t_spec, f_spec, Sxx_dB, shading="gouraud", cmap="magma")
    ax2.set_title("Spektrogram")
    ax2.set_xlabel("Czas [s]")
    ax2.set_ylabel("Częstotliwość [Hz]")

    max_expected_hz = max(hz, hz2) if signal_type == "Superpozycja (sin + cos)" else hz
    if signal_type in ["Prostokątny", "Piłokształtny", "Impuls jednostkowy"]:
        ax2.set_ylim(0, fs / 2)
    else:
        ax2.set_ylim(0, max_expected_hz + 200)

    fig2.colorbar(im, ax=ax2, label="Moc sygnału [dB]")

    plt.tight_layout()
    c2 = st.container()
    with c2:
        st.pyplot(fig2)

    st.subheader("Eksport danych")

    df = pd.DataFrame({"Czas": t, "Amplituda": y})

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Pobierz sygnał jako CSV",
        data=csv_data,
        file_name=f"sygnal_{signal_type.lower().replace(' ', '_')}_{hz}Hz.csv",
        mime="text/csv",
    )


def task_2():
    f0 = st.slider("Częstotliwość początkowa (Hz)", 1, 20, 5, key="emd_f0")
    f1 = st.slider("Częstotliwość końcowa (Hz)", 50, 150, 100, key="emd_f1")

    fs = 1000
    t = np.linspace(0, 3, fs, endpoint=False)
    signal = sig.chirp(t, f0=f0, f1=f1, t1=1, method="linear")

    imfs = emd.sift.sift(signal)
    num_imfs = imfs.shape[1]

    xf = fft.fftfreq(fs, 1 / fs)[: fs // 2]
    max_hz = max(f0, f1) + 20

    c_orig = st.container()
    with c_orig:
        st.subheader(f"Chirp {f0} Hz - {f1} Hz")
        fig_orig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 2))
        ax_t.plot(t, signal, color="black", linewidth=1)
        ax_t.set_title("Dziedzina czasu")
        ax_t.set_ylabel("Amplituda")
        ax_t.set_xlabel("Czas [s]")
        ax_t.grid(True, linestyle="--", alpha=0.5)
        ax_t.set_xlim(0, 1)

        yf = fft.fft(signal)
        ax_f.plot(xf, 2.0 / fs * np.abs(yf[: fs // 2]), color="black", linewidth=1)
        ax_f.set_title("Widmo amplitudowe")
        ax_f.set_ylabel("Amplituda")
        ax_f.set_xlabel("Częstotliwość [Hz]")
        ax_f.grid(True, linestyle="--", alpha=0.5)
        ax_f.set_xlim(0, max_hz)

        plt.tight_layout()
        st.pyplot(fig_orig)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i in range(num_imfs):
        c_imf = st.container()
        with c_imf:
            st.subheader(f"Mode {i+1}")

            fig_imf, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 2))

            color = colors[i % len(colors)]
            imf = imfs[:, i]

            ax_t.plot(t, imf, color=color, linewidth=1.2)
            ax_t.set_title("Dziedzina czasu")
            ax_t.set_ylabel("Amplituda")
            ax_t.set_xlabel("Czas [s]")
            ax_t.grid(True, linestyle="--", alpha=0.5)
            ax_t.set_xlim(0, 1)

            yf_imf = fft.fft(imf)
            ax_f.plot(
                xf, 2.0 / fs * np.abs(yf_imf[: fs // 2]), color=color, linewidth=1.2
            )
            ax_f.set_title("Widmo amplitudowe")
            ax_f.set_ylabel("Amplituda")
            ax_f.set_xlabel("Częstotliwość [Hz]")
            ax_f.grid(True, linestyle="--", alpha=0.5)
            ax_f.set_xlim(0, max_hz)

            plt.tight_layout()
            st.pyplot(fig_imf)


def task_3():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Składnik 1**")
        active1 = True
        type1 = st.selectbox("Typ 1", ["Sinus", "Cosinus"], key="t1")
        f1 = st.slider("Częstotliwość (Hz)", 1, 100, 5, key="f1")
        a1 = st.slider("Amplituda", 0.1, 5.0, 2.0, key="a1")

    with col2:
        st.markdown("**Składnik 2**")
        active2 = True
        type2 = st.selectbox("Typ 2", ["Sinus", "Cosinus"], key="t2")
        f2 = st.slider("Częstotliwość (Hz)", 1, 100, 25, key="f2")
        a2 = st.slider("Amplituda", 0.1, 5.0, 1.0, key="a2")

    fs = 1000
    t_max = 3.0
    t = np.linspace(0, t_max, int(fs * t_max), endpoint=False)
    signal = np.zeros_like(t)
    max_hz_used = 100

    if active1:
        signal += a1 * (
            np.sin(2 * np.pi * f1 * t)
            if type1 == "Sinus"
            else np.cos(2 * np.pi * f1 * t)
        )
    if active2:
        signal += a2 * (
            np.sin(2 * np.pi * f2 * t)
            if type2 == "Sinus"
            else np.cos(2 * np.pi * f2 * t)
        )

    imfs = emd.sift.sift(signal)
    num_imfs = imfs.shape[1]

    N = len(signal)
    xf = fft.fftfreq(N, 1 / fs)[: N // 2]
    max_hz_plot = max_hz_used + 20

    c_orig = st.container()
    with c_orig:
        st.subheader(f"Superpozycja {type1} {f1} Hz + {type2} {f2} Hz")
        fig_orig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 4))

        ax_t.plot(t, signal, color="black", linewidth=1)
        ax_t.set_title("Dziedzina czasu")
        ax_t.set_ylabel("Amplituda")
        ax_t.set_xlabel("Czas [s]")
        ax_t.grid(True, linestyle="--", alpha=0.5)
        ax_t.set_xlim(0, t_max)

        yf = fft.fft(signal)
        ax_f.plot(xf, 2.0 / N * np.abs(yf[: N // 2]), color="black", linewidth=1)
        ax_f.set_title("Widmo amplitudowe (Powinny być widoczne piki)")
        ax_f.set_ylabel("Amplituda")
        ax_f.set_xlabel("Częstotliwość [Hz]")
        ax_f.grid(True, linestyle="--", alpha=0.5)
        ax_f.set_xlim(0, max_hz_plot)

        plt.tight_layout()
        st.pyplot(fig_orig)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i in range(num_imfs):
        c_imf = st.container()
        with c_imf:
            st.subheader(f"Mode {i+1}")
            fig_imf, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 2))

            color = colors[i % len(colors)]
            imf = imfs[:, i]

            ax_t.plot(t, imf, color=color, linewidth=1.2)
            ax_t.set_title("Dziedzina czasu")
            ax_t.set_ylabel("Amplituda")
            ax_t.set_xlabel("Czas [s]")
            ax_t.grid(True, linestyle="--", alpha=0.5)
            ax_t.set_xlim(0, t_max)

            yf_imf = fft.fft(imf)
            ax_f.plot(
                xf, 2.0 / N * np.abs(yf_imf[: N // 2]), color=color, linewidth=1.2
            )
            ax_f.set_title("Widmo amplitudowe")
            ax_f.set_ylabel("Amplituda")
            ax_f.set_xlabel("Częstotliwość [Hz]")
            ax_f.grid(True, linestyle="--", alpha=0.5)
            ax_f.set_xlim(0, max_hz_plot)

            plt.tight_layout()
            st.pyplot(fig_imf)


def task_4():
    uploaded_file = st.file_uploader("Wybierz plik z sygnałem (.csv)", type=["csv"])

    if uploaded_file is not None:
        try:

            df = pd.read_csv(uploaded_file)

            if "Czas" in df.columns and "Amplituda" in df.columns:
                t = df["Czas"].values
                signal = df["Amplituda"].values
            else:
                t = df.iloc[:, 0].values
                signal = df.iloc[:, 1].values

            dt = t[1] - t[0]
            fs = 1.0 / dt
            N = len(signal)

            max_fft_freq = 100

            imfs = emd.sift.sift(signal)
            num_imfs = imfs.shape[1]

            xf = fft.fftfreq(N, dt)[: N // 2]

            c_orig = st.container()
            with c_orig:
                st.subheader(f"Sygnał z pliku: {uploaded_file.name}")
                fig_orig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 2))

                ax_t.plot(t, signal, color="black", linewidth=1)
                ax_t.set_title("Dziedzina czasu")
                ax_t.set_ylabel("Amplituda")
                ax_t.set_xlabel("Czas [s]")
                ax_t.grid(True, linestyle="--", alpha=0.5)
                ax_t.set_xlim(t[0], t[-1])

                yf = fft.fft(signal)
                ax_f.plot(
                    xf, 2.0 / N * np.abs(yf[: N // 2]), color="black", linewidth=1
                )
                ax_f.set_title("Widmo amplitudowe")
                ax_f.set_ylabel("Amplituda")
                ax_f.set_xlabel("Częstotliwość [Hz]")
                ax_f.grid(True, linestyle="--", alpha=0.5)
                ax_f.set_xlim(0, max_fft_freq)

                plt.tight_layout()
                st.pyplot(fig_orig)

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

            for i in range(num_imfs):
                c_imf = st.container()
                with c_imf:
                    st.subheader(f"Mode {i+1}")
                    fig_imf, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 2))

                    color = colors[i % len(colors)]
                    imf = imfs[:, i]

                    ax_t.plot(t, imf, color=color, linewidth=1.2)
                    ax_t.set_title("Dziedzina czasu")
                    ax_t.set_ylabel("Amplituda")
                    ax_t.set_xlabel("Czas [s]")
                    ax_t.grid(True, linestyle="--", alpha=0.5)
                    ax_t.set_xlim(t[0], t[-1])

                    yf_imf = fft.fft(imf)
                    ax_f.plot(
                        xf,
                        2.0 / N * np.abs(yf_imf[: N // 2]),
                        color=color,
                        linewidth=1.2,
                    )
                    ax_f.set_title("Widmo amplitudowe")
                    ax_f.set_ylabel("Amplituda")
                    ax_f.set_xlabel("Częstotliwość [Hz]")
                    ax_f.grid(True, linestyle="--", alpha=0.5)
                    ax_f.set_xlim(0, max_fft_freq)

                    plt.tight_layout()
                    st.pyplot(fig_imf)

        except Exception as e:
            st.error(
                f"Wystąpił błąd podczas analizy pliku. Upewnij się, że to poprawny plik CSV z danymi liczbowymi. Szczegóły błędu: {e}"
            )


tasks_details = {
    1: {
        "title": "Zadanie 1: Spektogramy",
        "description": "Przygotuj kod w Pythonie, który wygeneruje spektrogramy dla sygnałów z zadania 1 na liście 1.",
        "func": task_1,
    },
    2: {
        "title": "Zadanie 2: Dekompozycja oraz widmo",
        "description": "Przygotuj w Pythonie kod bazując na pakiecie emd, który dokona dekompozycji sygnału świergotliwego (chirp) oraz wyznaczy widmo każdej mody. ",
        "func": task_2,
    },
    3: {
        "title": "Zadanie 3: Dekompozycja superpozycji",
        "description": "Przygotuj w Pythonie kod bazując na pakiecie emd, który dokona dekompozycji dowolnie zbudowanego sygnału będącego superpozycją kilku funkcji sinus i cosinus. Wyznacz widmo każdej z mod. ",
        "func": task_3,
    },
    4: {
        "title": "Zadanie 4: Dekompozycja sygnału z csv",
        "description": "Przygotuj w Pythonie kod bazując na pakiecie emd, który dokona dekompozycji sygnału załadowanego z pliku np. csv. Wyznacz widmo każdej z mod. ",
        "func": task_4,
    },
}
