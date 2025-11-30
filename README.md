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
- XGBoost </br>
<img width="639" height="456" alt="image" src="https://github.com/user-attachments/assets/ad0a93ef-f09e-4e06-91e7-24c6ad1de90e" /> </br>
<img width="428" height="415" alt="image" src="https://github.com/user-attachments/assets/32d49099-c51c-40c9-89be-53e426dc4e4c" /> </br>
<img width="524" height="449" alt="image" src="https://github.com/user-attachments/assets/3bc5dd21-16e1-4f6d-8b68-c74528be4850" /> </br>




- Drzewo decyzyjne (wariant 1) </br>

<img width="585" height="515" alt="image" src="https://github.com/user-attachments/assets/98087029-54cd-4a6d-9199-63be53163b9d" /> </br>
<img width="595" height="565" alt="image" src="https://github.com/user-attachments/assets/c7d79eeb-fb55-492a-956e-0f6caae60f12" /> </br>

- Drzewo decyzyjne (wariant 2) </br>

<img width="491" height="317" alt="image" src="https://github.com/user-attachments/assets/4a814dfa-acd4-4446-a336-e606ffb6a8ae" /> </br>
<img width="1504" height="489" alt="image" src="https://github.com/user-attachments/assets/03937e27-95b3-4b7f-94f0-b3410d0debd5" /> </br>

---
## Wyniki klasyfikacji 

<img width="912" height="111" alt="image" src="https://github.com/user-attachments/assets/377850b5-e163-4231-8879-e709054a4e27" />

---

### Klasteryzacja

Testowane algorytmy:
- KMeans  
- MiniBatchKMeans  
- Gaussian Mixture Models (GMM)  
- BIRCH  

---

