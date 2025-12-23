
# 🏃‍♂️ Halfmarathon Predictor – Aplikacja Webowa

> **Sprawdź działającą aplikację tutaj:**
> ### 👉 [Otwórz Halfmarathon Predictor (Live Demo)](https://monkfish-app-tfxue.ondigitalocean.app/)

![Status Projektu](https://img.shields.io/badge/Status-Deployed-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

## 📄 O Projekcie
Halfmarathon Predictor to narzędzie oparte na uczeniu maszynowym, które pomaga biegaczom oszacować ich potencjalny czas ukończenia półmaratonu.

Zamiast zgadywać tempo startowe, użytkownik może wprowadzić swoje parametry treningowe i otrzymać predykcję opartą na danych historycznych.

### 🔴 Problem
Wielu amatorów biegania ma trudności z dobraniem odpowiedniej strategii na start.
* Zbyt optymistyczne założenia kończą się "ścianą" na 15. kilometrze.
* Zbyt zachowawczy bieg to strata szansy na rekord życiowy (PB).

### 🟢 Rozwiązanie
Stworzyłem model regresji, który analizuje kluczowe czynniki wpływające na wydolność i przewiduje czas końcowy z wysoką dokładnością. Aplikacja udostępnia ten model w formie prostego interfejsu webowego.

---

## ⚙️ Jak to działa?
Model analizuje dane wejściowe, takie jak:
* Wiek, 
* Płeć 
* Twój dystczasans na 5 km.


Na podstawie tych danych model zwraca przewidywany czas (np. `01:45:30`) oraz sugerowane średnie tempo biegu (`5:00 min/km`).

---

## 🛠️ Stack Technologiczny

### Machine Learning (Backend)
* **Python & Pandas:** Czyszczenie danych i inżynieria cech.
* **Scikit-Learn:** Trenowanie modelu (np. Random Forest Regressor / Linear Regression).
* **Pickle:** Serializacja modelu do pliku, aby mógł być użyty w aplikacji.

### Web & Deployment
* **Streamlit:** Framework do budowy interfejsu użytkownika (Front-end) w czystym Pythonie.
* **DigitalOcean App Platform:** Hosting i wdrożenie aplikacji w chmurze (CI/CD z GitHub).

---

## 🖥️ Jak uruchomić lokalnie?
Jeśli chcesz przetestować kod na własnym komputerze:

1.  Sklonuj repozytorium:
    ```bash
    git clone [https://github.com/MichalBorek1983/Portfolio.git](https://github.com/MichalBorek1983/Portfolio.git)
    ```
2.  Przejdź do folderu projektu:
    ```bash
    cd "ds_ai_portfolio/docs/Halfmarathon Predictor"
    ```
3.  Zainstaluj zależności:
    ```bash
    pip install -r requirements.txt
    ```
4.  Uruchom aplikację:
    ```bash
    streamlit run app.py
    ```

---
*Autor: Michał Borek*