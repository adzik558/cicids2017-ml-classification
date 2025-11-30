# CICIDS2017 – Klasyfikacja ataków sieciowych (XGBoost, Drzewa decyzyjne, Klasteryzacja)

---

## Opis projektu

Celem projektu jest analiza zachowań ruchu sieciowego i wykrywanie ataków przy użyciu klasycznych metod machine learning.  
Projekt został wykonany na bazie rzeczywistego zbioru **CICIDS2017**, zawierającego:  
- obserwacje normalnego ruchu,  
- ataki DDoS, DoS, Brute Force, Infiltration, Botnet, Web Attacks,  
- ponad **80 cech numerycznych**.



---

### Źródło danych

- **CICIDS2017** – Canadian Institute for Cybersecurity  
- Link: https://www.kaggle.com/datasets/cicdataset/cicids2017 

Ze względu na rozmiar (setki MB / miliony wierszy) analizy wykonywane były na próbkach (30–50%), co dokładnie opisano w dokumentacji PDF.

---

Ten projekt stanowi kompletną analizę zbioru **CICIDS2017**, obejmującą:
- przygotowanie danych,
- eksplorację i analizę statystyczną,
- klasyfikację ataków (XGBoost, Drzewo Decyzyjne),
- redukcję wymiarów (PCA),
- analizę klasteryzacji (KMeans, MiniBatchKMeans, GMM, BIRCH),
- wizualizacje oraz wnioski.

Pełna dokumentacja projektu znajduje się w:  
`docs/klasteryzacja_xgboost_drzewa.pdf`

---

## Struktura repozytorium

```plaintext
cicids2017-ml-classification/
├── README.md
├── LICENSE
├── docs/
│   └── dokumentacja - demografia.pdf
├── src/
│   └── analysis.R
└── data/
    └── tablice_trwania_zycia_w_latach_1990-2022.xlsx
```

---

### Klasyfikacja

W projekcie użyto:
- XGBoost
- Drzewo decyzyjne (2 warianty)

---

### Klasteryzacja

Testowane algorytmy:
- KMeans  
- MiniBatchKMeans  
- Gaussian Mixture Models (GMM)  
- BIRCH  

---

