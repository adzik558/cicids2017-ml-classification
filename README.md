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

Ze względu na rozmiar (setki MB / miliony wierszy) analizy wykonywane były na próbkach (30–50%)
---

Ten projekt stanowi kompletną analizę zbioru **CICIDS2017**, obejmującą:
- przygotowanie danych,
- eksplorację i analizę statystyczną,
- klasyfikację ataków (XGBoost, Drzewo Decyzyjne),
- redukcję wymiarów (PCA),
- analizę klasteryzacji (KMeans, MiniBatchKMeans, GMM, BIRCH),
- wizualizacje oraz wnioski.

---

## Struktura repozytorium

```plaintext
cicids2017-ml-classification/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   └── clustering-and-classification-attack-types.ipynb
└── data/
    └── data
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

<img width="474" height="453" alt="image" src="https://github.com/user-attachments/assets/f96ee89a-462a-4b00-a23e-f0f632d09de9" /> </br>



- Drzewo decyzyjne  </br>

<img width="474" height="453" alt="image" src="https://github.com/user-attachments/assets/e4657310-f05d-4f90-a58e-69644863ce77" /> </br>


---
## Wyniki klasyfikacji 

XGBoost - błędne klasyfikacje: </br>
```
| Prawdziwa klasa ↓ | Predykcja →      | Liczba błędów |
| ----------------- | ---------------- | ------------- |
| **BENIGN**        | 1 (Bot)          | 32            |
|                   | 2 (DDoS)         | 8             |
|                   | 3 (DoS GE)       | 2             |
|                   | 4 (DoS Hulk)     | 81            |
|                   | 5 (Slowhttptest) | 11            |
|                   | 6 (slowloris)    | 8             |
|                   | 9 (PortScan)     | 65            |
| **Bot**           | 0 (BENIGN)       | 61            |
| **DDoS**          | 0 (BENIGN)       | 12            |
|                   | 5 (Slowhttptest) | 1             |
| **DoS GoldenEye** | 0 (BENIGN)       | 3             |
|                   | 4 (DoS Hulk)     | 2             |
| **DoS Hulk**      | 0 (BENIGN)       | 35            |
|                   | 2 (DDoS)         | 2             |
|                   | 3 (DoS GE)       | 1             |
|                   | 9 (PortScan)     | 1             |
| **Slowhttptest**  | 0 (BENIGN)       | 21            |
|                   | 3 (DoS GE)       | 20            |
| **slowloris**     | 0 (BENIGN)       | 11            |
|                   | 4 (DoS Hulk)     | 1             |
| **Heartbleed**    | 0 (BENIGN)       | 5             |
|                   | 6 (slowloris)    | 1             |
| **Infiltration**  | 0 (BENIGN)       | 1             |
| **PortScan**      | 0 (BENIGN)       | 5             |
|                   | 3 (DoS GE)       | 4             |
```
</br></br>

Drzewo decyzyjne - błedne klasyfikacje: </br>
```
| Prawdziwa klasa ↓ | Predykcja →  | Liczba błędów |
| ----------------- | ------------ | ------------- |
| **BENIGN**        | 1 (Bot)      | 32            |
|                   | 2 (DDoS)     | 2             |
|                   | 3 (DoS GE)   | 8             |
|                   | 4 (DoS Hulk) | 26            |
|                   | 5 (Slowhttp) | 11            |
|                   | 6 (slowl.)   | 8             |
|                   | 9 (PortScan) | 64            |
| **Bot**           | 0 (BENIGN)   | 32            |
| **DDoS**          | 0 (BENIGN)   | 3             |
|                   | 1 (Bot)      | 1             |
|                   | 4 (DoS Hulk) | 1             |
| **DoS GoldenEye** | 0 (BENIGN)   | 3             |
|                   | 5 (Slowhttp) | 2             |
| **DoS Hulk**      | 0 (BENIGN)   | 15            |
|                   | 2 (DDoS)     | 2             |
|                   | 3 (DoS GE)   | 1             |
|                   | 9 (PortScan) | 1             |
| **Slowhttptest**  | 0 (BENIGN)   | 9             |
| **slowloris**     | 0 (BENIGN)   | 3             |
|                   | 4 (DoS Hulk) | 1             |
| **Heartbleed**    | 6 (slowl.)   | 1             |
| **Infiltration**  | 0 (BENIGN)   | 1             |
| **PortScan**      | 0 (BENIGN)   | 4             |
|                   | 3 (DoS GE)   | 4             |
```
</br></br></br>
---

## Tabela trafności przewidywań XGBoost i Drzewa decyzyjnego: </br>

<img width="1198" height="103" alt="image" src="https://github.com/user-attachments/assets/ddbb3630-b5a8-4ae2-8564-dd90e0be75d0" />

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


## Jak uruchomić
1. Zainstaluj R i RStudio
2. Zainstaluj pakiety:
install.packages(c("readxl","ggplot2","plotrix","dplyr"))
3. Pobierz data/tablice_trwania_zycia.csv
4. Pobierz src/demographic-analysis.R
5. Otwórz plik demographic-analysis.R
6. Zainstaluj wymagane biblioteki
7. Wybierz ścieżkę danych (linijki od 3 do 13 w kodzie src/demographic-analysis.R
8. Skompiluj kod
9. Wyniki oraz wykresy zostaną wyświetlone w interfejsie R.
