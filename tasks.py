import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import scipy.signal as sig
import emd
import pandas as pd


def random_code():
    x = st.slider("Liczba wyników do wygenerowania", 1, 10, 5)
    return (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def task_1():
    c1, c2 = st.columns(2)
    with c1:
        amplitude = st.slider("Amplituda sygnału bazowego", 1.0, 10.0, 5.0, step=0.5)
        frequency = st.slider("Częstotliwość (Hz)", 1, 20, 5)
    with c2:
        noise_std = st.slider(
            "Poziom szumu (Odchylenie standardowe)", 0.0, 5.0, 1.0, step=0.1
        )

    fs = 1000
    t = np.linspace(0, 3, fs, endpoint=False)

    s = amplitude * np.sin(2 * np.pi * frequency * t)

    n = noise_std * np.random.randn(len(t))

    y = s + n

    N = len(s)

    mse = np.sum((s - y) ** 2) / N

    if mse == 0:
        snr = float("inf")
        psnr = float("inf")
    else:

        rms_s = np.sqrt(np.mean(s**2))
        rms_n = np.sqrt(np.mean(n**2))
        snr = 20 * np.log10(rms_s / rms_n)

        s_max = np.max(np.abs(s))
        psnr = 20 * np.log10(s_max / np.sqrt(mse))

    st.subheader("Obliczone miary:")
    col_m1, col_m2, col_m3 = st.columns(3)

    col_m1.metric("MSE (Błąd średniokwadratowy)", f"{mse:.4f}")
    col_m2.metric("SNR (Stosunek sygnał/szum)", f"{snr:.2f} dB")
    col_m3.metric("PSNR (Szczytowy SNR)", f"{psnr:.2f} dB")

    st.divider()
    st.subheader("Wizualizacja zniekształceń")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        t,
        y,
        color="#d62728",
        alpha=0.7,
        linewidth=1.5,
        label="Sygnał z zakłóceniem (y)",
    )

    ax.plot(t, s, color="black", linewidth=2, label="Oryginalny sygnał (s)")

    ax.set_title("Porównanie sygnału czystego z zaszumionym")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 3)

    plt.tight_layout()
    c_plot = st.container()
    with c_plot:
        st.pyplot(fig)


def task_2():
    c1, c2 = st.columns(2)
    with c1:
        amplitude = st.slider(
            "Amplituda sygnału", 1.0, 10.0, 5.0, step=0.5, key="t2_amp"
        )
        frequency = st.slider("Częstotliwość (Hz)", 1, 20, 5, key="t2_freq")
    with c2:
        noise_std = st.slider("Poziom szumu", 0.0, 5.0, 1.0, step=0.1, key="t2_noise")

    fs = 1000
    t = np.linspace(0, 3, fs, endpoint=False)
    s = amplitude * np.sin(2 * np.pi * frequency * t)
    n = noise_std * np.random.randn(len(t))
    y = s + n

    N = len(s)
    mse_manual = np.sum((s - y) ** 2) / N

    if mse_manual == 0:
        psnr_manual = float("inf")
        snr_manual = float("inf")
    else:
        s_max_manual = np.max(np.abs(s))
        psnr_manual = 20 * np.log10(s_max_manual / np.sqrt(mse_manual))

        rms_s = np.sqrt(np.mean(s**2))
        rms_n = np.sqrt(np.mean(n**2))
        snr_manual = 20 * np.log10(rms_s / rms_n)

    mse_lib = mean_squared_error(s, y)
    s_max_lib = np.max(np.abs(s))
    psnr_lib = peak_signal_noise_ratio(s, y, data_range=s_max_lib)

    snr_lib = "Brak funkcji"

    col1, col2, col3 = st.columns(3)
    st.table(
        {
            "Miara": ["MSE", "PSNR", "SNR"],
            "Nasze wzory (Zadanie 1)": [
                f"{mse_manual:.6f}",
                f"{psnr_manual:.4f} dB",
                f"{snr_manual:.4f} dB",
            ],
            "Gotowe biblioteki (Python)": [
                f"{mse_lib:.6f}",
                f"{psnr_lib:.4f} dB",
                snr_lib,
            ],
        }
    )


def task_3():
    c1, c2 = st.columns(2)
    with c1:
        target_snr = st.slider("Docelowy SNR (dB)", -20.0, 40.0, 10.0, step=1.0)
        f0 = st.slider("Częstotliwość początkowa Chirp (Hz)", 1, 50, 5)
        f1 = st.slider("Częstotliwość końcowa Chirp (Hz)", 1, 100, 10)
    with c2:
        noise_type = st.selectbox(
            "Rodzaj szumu",
            ["Szum Biały", "Szum Browna"],
        )
    duration = 3

    fs = 2000
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    s = sig.chirp(t, f0=f0, f1=f1, t1=duration, method="linear")

    P_s = np.mean(s**2)

    P_n_target = P_s / (10 ** (target_snr / 10))

    if noise_type == "Szum Biały":
        n_raw = np.random.randn(len(t))
    else:
        n_raw = np.cumsum(np.random.randn(len(t)))
        n_raw = n_raw - np.mean(n_raw)

    P_n_raw = np.mean(n_raw**2)
    k = np.sqrt(P_n_target / P_n_raw)

    n_scaled = n_raw * k
    y = s + n_scaled

    actual_snr = 10 * np.log10(np.mean(s**2) / np.mean(n_scaled**2))

    st.write(f"Obliczony SNR: {actual_snr:.2f} dB")

    st.divider()

    c_time = st.container()
    with c_time:
        st.subheader("Sygnał z zakłóceniem")
        fig_t, ax_t = plt.subplots(figsize=(10, 2))

        display_samples = int(3 * fs)
        ax_t.plot(
            t[:display_samples],
            y[:display_samples],
            color="#d62728",
            label="Chirp + Szum",
            linewidth=1.5,
            alpha=0.8,
        )
        ax_t.plot(
            t[:display_samples],
            s[:display_samples],
            color="black",
            label="Chirp",
            linewidth=1,
        )

        ax_t.set_xlabel("Czas [s]")
        ax_t.set_ylabel("Amplituda")
        ax_t.legend(loc="upper right")
        ax_t.grid(True, linestyle="--", alpha=0.5)
        ax_t.set_xlim(0, 3)

        plt.tight_layout()
        st.pyplot(fig_t)

    c_spec = st.container()
    with c_spec:
        st.subheader("Spektrogram")
        fig_s, ax_s = plt.subplots(figsize=(10, 2))

        f_spec, t_spec, Sxx = sig.spectrogram(y, fs, nperseg=256, noverlap=128)
        Sxx_dB = 10 * np.log10(Sxx + 1e-10)

        im = ax_s.pcolormesh(t_spec, f_spec, Sxx_dB, shading="gouraud", cmap="magma")
        ax_s.set_title(f"Spektrogram - {noise_type}")
        ax_s.set_xlabel("Czas [s]")
        ax_s.set_ylabel("Częstotliwość [Hz]")
        ax_s.set_ylim(0, 120)

        fig_s.colorbar(im, ax=ax_s, label="Moc [dB]")

        plt.tight_layout()
        st.pyplot(fig_s)


def task_4():
    c1, c2 = st.columns(2)
    with c1:
        target_snr = st.slider("Początkowy SNR (dB)", -10.0, 30.0, 5.0, step=1.0)
        f0 = st.slider("Częstotliwość pocz. Chirp (Hz)", 1, 50, 5)
    with c2:
        noise_type = st.selectbox(
            "Rodzaj szumu",
            ["Szum Biały (Płaskie widmo)", "Szum Browna (Niskie częstotliwości)"],
        )
        f1 = st.slider("Częstotliwość końc. Chirp (Hz)", 1, 50, 10)

    st.subheader("Parametry Filtru Wienera")
    window_size = st.slider(
        "Rozmiar okna filtru (w próbkach)",
        min_value=3,
        max_value=201,
        value=29,
        step=2,
        help="Im większe okno, tym silniejsze wygładzanie, ale większe ryzyko zniekształcenia oryginalnego sygnału.",
    )

    fs = 1000
    duration = 3
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    s = sig.chirp(t, f0=f0, f1=f1, t1=duration, method="linear")

    P_s = np.mean(s**2)
    P_n_target = P_s / (10 ** (target_snr / 10))

    if noise_type == "Szum Biały (Płaskie widmo)":
        n_raw = np.random.randn(len(t))
    else:
        n_raw = np.cumsum(np.random.randn(len(t)))
        n_raw = n_raw - np.mean(n_raw)

    P_n_raw = np.mean(n_raw**2)
    k = np.sqrt(P_n_target / P_n_raw)

    n_scaled = n_raw * k
    y_noisy = s + n_scaled

    y_filtered = sig.wiener(y_noisy, mysize=window_size)

    mse_before = np.mean((s - y_noisy) ** 2)
    snr_before = 10 * np.log10(np.mean(s**2) / np.mean((y_noisy - s) ** 2))

    mse_after = np.mean((s - y_filtered) ** 2)
    snr_after = 10 * np.log10(np.mean(s**2) / np.mean((y_filtered - s) ** 2))

    st.subheader("Ocena poprawy jakości")
    m1, m2 = st.columns(2)

    m1.metric(
        label="MSE (Błąd Średniokwadratowy)",
        value=f"{mse_after:.4f}",
        delta=f"{mse_after - mse_before:.4f}",
        delta_color="inverse",
    )

    m2.metric(
        label="SNR (Stosunek Sygnał/Szum)",
        value=f"{snr_after:.2f} dB",
        delta=f"{snr_after - snr_before:.2f} dB",
    )

    c_plots = st.container()
    with c_plots:

        fig1, ax1 = plt.subplots(figsize=(12, 2))

        display_samples = int(3 * fs)
        t_zoom = t[:display_samples]
        s_zoom = s[:display_samples]

        ax1.plot(
            t_zoom,
            y_noisy[:display_samples],
            color="#d62728",
            alpha=0.7,
            label="Sygnał zaszumiony",
        )
        ax1.plot(t_zoom, s_zoom, color="black", linewidth=1.5, label="Czysty oryginał")
        ax1.set_title("Sygnał zaszumiony")
        ax1.set_ylabel("Amplituda")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_xlim(0, 3)
        plt.tight_layout()
        st.pyplot(fig1)

    c_plots2 = st.container()
    with c_plots2:
        fig2, ax2 = plt.subplots(figsize=(12, 2))
        ax2.plot(
            t_zoom,
            y_filtered[:display_samples],
            color="#2ca02c",
            alpha=0.9,
            label="Przefiltrowany (Wiener)",
        )
        ax2.plot(
            t_zoom,
            s_zoom,
            color="black",
            linewidth=1.5,
            label="Czysty oryginał",
            linestyle="--",
        )
        ax2.set_title(f"Odszumiony Wienerem")
        ax2.set_xlabel("Czas [s]")
        ax2.set_ylabel("Amplituda")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.set_xlim(0, 3)

        plt.tight_layout()
        st.pyplot(fig2)


def task_5():
    c1, c2 = st.columns(2)
    with c1:
        target_snr = st.slider(
            "Początkowy SNR (dB)", -10.0, 30.0, 5.0, step=1.0, key="sg_snr"
        )
        f0 = st.slider("Częstotliwość pocz. Chirp (Hz)", 1, 50, 5, key="sg_f0")
    with c2:
        noise_type = st.selectbox(
            "Rodzaj szumu",
            ["Szum Biały (Płaskie widmo)", "Szum Browna (Niskie częstotliwości)"],
            key="sg_noise",
        )
        f1 = st.slider("Częstotliwość końc. Chirp (Hz)", 1, 50, 10, key="sg_f1")

    st.subheader("Parametry Filtru Savitzky'ego-Golaya")
    c3, c4 = st.columns(2)
    with c3:
        window_size = st.slider(
            "Rozmiar okna (window_length)",
            min_value=5,
            max_value=101,
            value=29,
            step=2,
            help="Liczba nieparzysta. Szersze okno to mocniejsze wygładzanie.",
        )
    with c4:
        max_poly = min(10, window_size - 1)
        poly_order = st.slider(
            "Rząd wielomianu (polyorder)",
            min_value=1,
            max_value=max_poly,
            value=3,
            help="Wyższy rząd lepiej zachowuje szczyty, ale gorzej tłumi szum. Zazwyczaj używa się 2 lub 3.",
        )

    fs = 1000
    duration = 3
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    s = sig.chirp(t, f0=f0, f1=f1, t1=duration, method="linear")

    P_s = np.mean(s**2)
    P_n_target = P_s / (10 ** (target_snr / 10))

    if noise_type == "Szum Biały (Płaskie widmo)":
        n_raw = np.random.randn(len(t))
    else:
        n_raw = np.cumsum(np.random.randn(len(t)))
        n_raw = n_raw - np.mean(n_raw)

    P_n_raw = np.mean(n_raw**2)
    k = np.sqrt(P_n_target / P_n_raw)

    n_scaled = n_raw * k
    y_noisy = s + n_scaled

    y_filtered = sig.savgol_filter(
        y_noisy, window_length=window_size, polyorder=poly_order
    )

    mse_before = np.mean((s - y_noisy) ** 2)
    snr_before = 10 * np.log10(np.mean(s**2) / np.mean((y_noisy - s) ** 2))

    mse_after = np.mean((s - y_filtered) ** 2)
    snr_after = 10 * np.log10(np.mean(s**2) / np.mean((y_filtered - s) ** 2))

    st.divider()
    st.subheader("Ocena poprawy jakości")

    m1, m2 = st.columns(2)

    m1.metric(
        label="MSE (Błąd Średniokwadratowy)",
        value=f"{mse_after:.4f}",
        delta=f"{mse_after - mse_before:.4f}",
        delta_color="inverse",
    )

    m2.metric(
        label="SNR (Stosunek Sygnał/Szum)",
        value=f"{snr_after:.2f} dB",
        delta=f"{snr_after - snr_before:.2f} dB",
    )

    c_plots = st.container()
    with c_plots:
        fig1, ax1 = plt.subplots(figsize=(12, 2))

        display_samples = int(3 * fs)
        t_zoom = t[:display_samples]
        s_zoom = s[:display_samples]

        ax1.plot(
            t_zoom,
            y_noisy[:display_samples],
            color="#d62728",
            alpha=0.7,
            label="Sygnał zaszumiony",
        )
        ax1.plot(t_zoom, s_zoom, color="black", linewidth=1.5, label="Czysty oryginał")
        ax1.set_title("Sygnał zaszumiony")
        ax1.set_ylabel("Amplituda")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_xlim(0, 3)
        plt.tight_layout()
        st.pyplot(fig1)

    c_noise = st.container()
    with c_noise:
        fig2, ax2 = plt.subplots(figsize=(12, 2))
        ax2.plot(
            t_zoom,
            y_filtered[:display_samples],
            color="#ff7f0e",
            alpha=0.9,
            label=f"Przefiltrowany (S-G, rząd {poly_order})",
        )
        ax2.plot(
            t_zoom,
            s_zoom,
            color="black",
            linewidth=1.5,
            label="Czysty oryginał",
            linestyle="--",
        )
        ax2.set_title(f"Odszumiony filtrem SG ")
        ax2.set_xlabel("Czas [s]")
        ax2.set_ylabel("Amplituda")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.set_xlim(0, 3)

        plt.tight_layout()
        st.pyplot(fig2)


def task_6():
    c1, c2 = st.columns(2)
    with c1:
        target_snr = st.slider(
            "Początkowy SNR (dB)", -10.0, 30.0, 5.0, step=1.0, key="emd_snr"
        )
        f0 = st.slider("Częstotliwość pocz. Chirp (Hz)", 1, 50, 5, key="emd_f0")
    with c2:
        noise_type = st.selectbox(
            "Rodzaj szumu",
            ["Szum Biały (Płaskie widmo)", "Szum Browna (Niskie częstotliwości)"],
            key="emd_noise",
        )
        f1 = st.slider("Częstotliwość końc. Chirp (Hz)", 1, 50, 10, key="emd_f1")

    fs = 1000
    duration = 3
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    s = sig.chirp(t, f0=f0, f1=f1, t1=duration, method="linear")

    P_s = np.mean(s**2)
    P_n_target = P_s / (10 ** (target_snr / 10))

    if noise_type == "Szum Biały (Płaskie widmo)":
        n_raw = np.random.randn(len(t))
    else:
        n_raw = np.cumsum(np.random.randn(len(t)))
        n_raw = n_raw - np.mean(n_raw)

    P_n_raw = np.mean(n_raw**2)
    k = np.sqrt(P_n_target / P_n_raw)

    n_scaled = n_raw * k
    y_noisy = s + n_scaled

    imfs = emd.sift.sift(y_noisy)
    num_imfs = imfs.shape[1]

    st.divider()
    st.subheader("Parametry Filtru EMD")

    discard_imfs = st.slider(
        "Liczba pierwszych modów (IMF) do odrzucenia",
        min_value=0,
        max_value=max(1, num_imfs - 1),
        value=1,
        help="IMF 1 to najszybsze drgania (najwięcej szumu). Zwiększaj tę wartość, aby wyrzucać kolejne warstwy szumu, ale uważaj, by nie wyciąć samego sygnału Chirp!",
    )

    if num_imfs > 1:
        y_filtered = np.sum(imfs[:, discard_imfs:], axis=1)
    else:
        y_filtered = y_noisy

    mse_before = np.mean((s - y_noisy) ** 2)
    snr_before = 10 * np.log10(np.mean(s**2) / np.mean((y_noisy - s) ** 2))

    mse_after = np.mean((s - y_filtered) ** 2)
    snr_after = 10 * np.log10(np.mean(s**2) / np.mean((y_filtered - s) ** 2))

    st.subheader("Ocena poprawy jakości")

    m1, m2 = st.columns(2)
    m1.metric(
        label="MSE (Błąd Średniokwadratowy)",
        value=f"{mse_after:.4f}",
        delta=f"{mse_after - mse_before:.4f}",
        delta_color="inverse",
    )
    m2.metric(
        label="SNR (Stosunek Sygnał/Szum)",
        value=f"{snr_after:.2f} dB",
        delta=f"{snr_after - snr_before:.2f} dB",
    )

    c_plots = st.container()
    with c_plots:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

        fig1, ax1 = plt.subplots(figsize=(12, 2))
        display_samples = int(3 * fs)
        t_zoom = t[:display_samples]
        s_zoom = s[:display_samples]

        ax1.plot(
            t_zoom,
            y_noisy[:display_samples],
            color="#d62728",
            alpha=0.7,
            label="Sygnał zaszumiony",
        )
        ax1.plot(t_zoom, s_zoom, color="black", linewidth=1.5, label="Czysty oryginał")
        ax1.set_title("Sygnał zaszumiony")
        ax1.set_ylabel("Amplituda")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_xlim(0, 3)
        plt.tight_layout()
        st.pyplot(fig1)

    c_plots2 = st.container()

    with c_plots2:
        fig2, ax2 = plt.subplots(figsize=(12, 2))
        ax2.plot(
            t_zoom,
            y_filtered[:display_samples],
            color="#9467bd",
            alpha=0.9,
            label=f"Przefiltrowany (Odrzucono {discard_imfs} IMF)",
        )
        ax2.plot(
            t_zoom,
            s_zoom,
            color="black",
            linewidth=1.5,
            label="Czysty oryginał",
            linestyle="--",
        )
        ax2.set_title(f"Częściowa rekonstrukcja EMD")
        ax2.set_xlabel("Czas [s]")
        ax2.set_ylabel("Amplituda")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.set_xlim(0, 3)

        plt.tight_layout()
        st.pyplot(fig2)

    st.divider()
    st.subheader("Eksport zaszumionego sygnału")

    df_noisy = pd.DataFrame({"Czas": t, "Amplituda": y_noisy})

    csv_noisy = df_noisy.to_csv(index=False).encode("utf-8")

    noise_prefix = "bialy" if "Biały" in noise_type else "browna"

    st.download_button(
        label="Pobierz zaszumiony sygnał jako CSV",
        data=csv_noisy,
        file_name=f"sygnal_zaszumiony_{noise_prefix}_{target_snr}dB.csv",
        mime="text/csv",
    )


def task_7():
    uploaded_file = st.file_uploader(
        "Wybierz plik z zaszumionym sygnałem (.csv)", type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "Czas" in df.columns and "Amplituda" in df.columns:
                t = df["Czas"].values
                y_noisy = df["Amplituda"].values
            else:
                t = df.iloc[:, 0].values
                y_noisy = df.iloc[:, 1].values

            dt = t[1] - t[0]
            fs = 1.0 / dt
            N = len(y_noisy)

            tab1, tab2, tab3 = st.tabs(
                ["Wiener", "Savitzky-Golay", "EMD (Dekompozycja)"]
            )

            with tab1:
                w_size = st.slider(
                    "Rozmiar okna (Wiener)", 3, 201, 29, step=2, key="w_size"
                )

                y_wiener = sig.wiener(y_noisy, mysize=w_size)

                fig1, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(
                    t,
                    y_noisy,
                    color="#d62728",
                    alpha=0.3,
                    label="Oryginał (Zaszumiony)",
                )
                ax1.plot(
                    t, y_wiener, color="#2ca02c", linewidth=2, label="Filtr Wienera"
                )
                ax1.set_title("Odszumianie Wienerem")
                ax1.set_xlabel("Czas [s]")
                ax1.set_ylabel("Amplituda")
                ax1.legend(loc="upper right")
                ax1.grid(True, linestyle="--", alpha=0.5)
                ax1.set_xlim(t[0], t[-1])
                st.pyplot(fig1)

            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    sg_size = st.slider(
                        "Rozmiar okna (S-G)", 5, 201, 51, step=2, key="sg_size"
                    )
                with c2:
                    sg_poly = st.slider(
                        "Rząd wielomianu", 1, min(10, sg_size - 1), 3, key="sg_poly"
                    )

                y_sg = sig.savgol_filter(
                    y_noisy, window_length=sg_size, polyorder=sg_poly
                )

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(
                    t,
                    y_noisy,
                    color="#d62728",
                    alpha=0.3,
                    label="Oryginał (Zaszumiony)",
                )
                ax2.plot(
                    t, y_sg, color="#ff7f0e", linewidth=2, label="Filtr Savitzky-Golay"
                )
                ax2.set_title("Odszumianie SG")
                ax2.set_xlabel("Czas [s]")
                ax2.set_ylabel("Amplituda")
                ax2.legend(loc="upper right")
                ax2.grid(True, linestyle="--", alpha=0.5)
                ax2.set_xlim(t[0], t[-1])
                st.pyplot(fig2)

            with tab3:
                cache_key = f"emd_imfs_{uploaded_file.name}"
                if cache_key not in st.session_state:
                    with st.spinner("Trwa dekompozycja EMD..."):
                        st.session_state[cache_key] = emd.sift.sift(y_noisy)

                imfs = st.session_state[cache_key]
                num_imfs = imfs.shape[1]

                discard = st.slider(
                    "Liczba początkowych modów do odrzucenia (Szum)",
                    0,
                    max(1, num_imfs - 1),
                    1,
                    key="emd_discard",
                )

                if num_imfs > 1:
                    y_emd = np.sum(imfs[:, discard:], axis=1)
                else:
                    y_emd = y_noisy

                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.plot(
                    t,
                    y_noisy,
                    color="#d62728",
                    alpha=0.3,
                    label="Oryginał (Zaszumiony)",
                )
                ax3.plot(
                    t,
                    y_emd,
                    color="#9467bd",
                    linewidth=2,
                    label=f"Rekonstrukcja EMD (Bez IMF 1-{discard})",
                )
                ax3.set_title("Odszumianie EMD")
                ax3.set_xlabel("Czas [s]")
                ax3.set_ylabel("Amplituda")
                ax3.legend(loc="upper right")
                ax3.grid(True, linestyle="--", alpha=0.5)
                ax3.set_xlim(t[0], t[-1])
                st.pyplot(fig3)

        except Exception as e:
            st.error(f"Błąd podczas analizy pliku CSV. Szczegóły: {e}")


tasks_details = {
    1: {
        "title": "Zadanie 1: Miary przeróżne",
        "description": """Przygotuj kod w Pythonie, które wyznacza wartości następujących miar jakości sygnałów: 
        SNR = 20log $$(\\frac{s}{n})$$, 
        PSNR = 20log $$(\\frac{s_{max}}{\\sqrt{MSE}})$$, 
        MSE = $$\\frac{1}{N}\\sum_{n=1}^{N}(s_n - y_n)$$, gdzie: s to sygnał, n to szum, $$s_{max}$$ to maksymalna wartość sygnału, y to sygnał z zakłóceniem.""",
        "func": task_1,
    },
    2: {
        "title": "Zadanie 2: Porównanie miar jakości",
        "description": """Przygotuj kod w Pythonie, który pozwoli na porównanie wartości miar SNR, PSNR, MSE 
        przygotowanych w zadaniu 1 oraz z gotowych implementacji dostępnych w języku Python.""",
        "func": task_2,
    },
    3: {
        "title": "Zadanie 3: Porównanie miar jakości",
        "description": """Przygotuj kod w Pythonie, który wygeneruje sygnał świergotliwy z szumem biały oraz szumem browna z zadanym SNR.""",
        "func": task_3,
    },
    4: {
        "title": "Zadanie 4: Filtr Wienera",
        "description": """Przygotuj kod w Pythonie, który od szumi sygnał z zadania 3 z wykorzystaniem filtru Wienera.""",
        "func": task_4,
    },
    5: {
        "title": "Zadanie 5: Filtr Savitzkyego-Golaya",
        "description": """Przygotuj kod w Pythonie, który od szumi sygnał z zadania 3 z wykorzystaniem filtru Savitzkyego-Golaya.""",
        "func": task_5,
    },
    6: {
        "title": "Zadanie 6: Filtr na bazie algorytmu EMD",
        "description": """Przygotuj kod w Pythonie, który od szumi sygnał z zadania 3 z wykorzystaniem filtru bazującego na algorytmie EMD i częściowej rekonstrukcji.""",
        "func": task_6,
    },
    7: {
        "title": "Zadanie 7: Odszumianie CSV",
        "description": """ Przygotuj kod w Pythonie, który od szumi sygnał załadowany z pliku csv oraz od szumi ten sygnał 
        wykorzystując filtr Wienera, Savitzkyego-Golaya oraz bazującego na algorytmie EMD i częściowej rekonstrukcji.""",
        "func": task_7,
    },
}
