import streamlit as st


def random_code():
    x = st.slider("Liczba wyników do wygenerowania", 1, 10, 5)
    return (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


tasks_details = {
    1: {
        "title": "Zadanie 1: Widmowa gęstość mocy",
        "description": "Dla sygnałów z zadania 1 z listy 1 napisz kod w języku Python, który wyznaczy dla nich widmową gęstość mocy. W tym celu wykorzystaj metody: periodogram i welcha. Metody dostępne są m.in. w bibliotece SciPy. ",
    },
    2: {
        "title": "Zadanie 2: WGM z definicji",
        "description": "Przygotuj w Pythonie kod, który wyznaczy z definicji widmową gęstość mocy. WGM z definicji wyznaczana jest z zależności: $$S_{xx}(f) = \int_{-\infty}^{\infty} R_{xx}(\\tau) e^{-i2\pi f\\tau} d\\tau$$",
    },
    3: {
        "title": "Zadanie 3: Porównanie metod",
        "description": "Dla sygnałów z zadania 1 z listy 1 napisz kod w języku Python, który pozwoli na porównanie wyników uzyskanych dla metod bibliotecznych (periodogram i welch) z samodzielną implementacją z definicji.",
        "func": random_code,
    },
}
