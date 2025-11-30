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


## Mapowanie etykiet (string → kod numeryczny)

| Kod | Etykieta              |
|-----|------------------------|
| 0   | BENIGN                |
| 1   | Bot                   |
| 2   | DDoS                  |
| 3   | DoS GoldenEye         |
| 4   | DoS Hulk              |
| 5   | DoS Slowhttptest      |
| 6   | DoS slowloris         |
| 7   | Heartbleed            |
| 8   | Infiltration          |
| 9   | PortScan              |

---
### Klasyfikacja

W projekcie użyto:
- XGBoost </br>

<img width="428" height="415" alt="image" src="https://github.com/user-attachments/assets/32d49099-c51c-40c9-89be-53e426dc4e4c" /> </br>



- Drzewo decyzyjne (wariant 1) </br>

<img width="585" height="515" alt="image" src="https://github.com/user-attachments/assets/98087029-54cd-4a6d-9199-63be53163b9d" /> </br>


---
## Wyniki klasyfikacji 

<img width="912" height="111" alt="image" src="https://github.com/user-attachments/assets/377850b5-e163-4231-8879-e709054a4e27" />

---

### Klasteryzacja

Testowane algorytmy:
- KMeans </br>

<img width="993" height="671" alt="image" src="https://github.com/user-attachments/assets/a025c908-1718-49ae-ad65-4e4ac5de0f16" /> </br>

- MiniBatchKMeans </br>

<img width="908" height="612" alt="image" src="https://github.com/user-attachments/assets/6e2bc4df-8f65-4422-a038-0645a2426712" /> </br>

- Gaussian Mixture Models (GMM) </br>

<img width="1012" height="685" alt="image" src="https://github.com/user-attachments/assets/c082edb2-195e-4393-821f-d6fd3feda0ce" /> </br>

- BIRCH </br>

<img width="1010" height="605" alt="image" src="https://github.com/user-attachments/assets/10b83b4f-82ae-45cd-b6ef-0db8e5193cb0" /> </br>

## Wyniki klasteryzacji
<img width="745" height="682" alt="image" src="https://github.com/user-attachments/assets/04f91b72-03d7-4c4c-bf0e-836598cd46d8" /> </br>

<img width="752" height="115" alt="image" src="https://github.com/user-attachments/assets/daa51061-feee-401a-8a81-38be7f9d3d04" />


---
### Podsumowanie
Główną różnicą pomiędzy klasyfikacją a klasteryzacją jest typ uczenia: </br>
- Klasyfikacja jest zadaniem uczenia nadzorowanego, gdzie proces uczenia odbywa się na
podstawie danych uczących z etykietami lub prawidłowymi odpowiedziami.
- Klasteryzacja jest zadaniem uczenia nienadzorowanego, co oznacza, że nie ma dostępnych
etykiet ani prawidłowych odpowiedzi w danych uczących. Celem klasteryzacji jest odkrycie
struktury ukrytej w danych poprzez grupowanie podobnych obiektów.

Ponadto: </br>
- XGBoost jest zdecydowanie najlepszym modelem dla CICIDS2017.
- Drzewo decyzyjne zachowuje się świetnie na niektórych próbkach (nawet 0.99 accuracy), ale znacznie gorzej na innych → podatność na overfitting.
- Klasteryzacja wykazuje realne segmenty ruchu (silhouette ~0.77 dla KMeans).
- Dane wymagają intensywnego czyszczenia (odstające wartości, brakujące dane).

