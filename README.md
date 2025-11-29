# Klasyfikacja ataków sieciowych w zbiorze CICIDS2017 (Python, ML)

Projekt dotyczy analizy zbioru danych **CICIDS2017**, który zawiera ruch sieciowy
zawierający zarówno normalne połączenia, jak i różne typy ataków.
Celem pracy było przygotowanie danych, budowa modeli klasyfikacyjnych oraz
przetestowanie metod klasteryzacji w zadaniu wykrywania zagrożeń.

W projekcie wykorzystano modele nadzorowane (XGBoost, Drzewo decyzyjne)
oraz nienadzorowane (K-Means, MiniBatch K-Means, GMM, BIRCH).

---

## Pliki znajdujące się w repozytorium
- **Klasteryzacja_XGBoost_i_Drzewa_Decyzyjne.ipynb** – kompletny notebook z kodem analizy  
- **Klasteryzacja_XGBoost_i_Drzewa_Decyzyjne.pdf** – pełna dokumentacja projektu (raport akademicki)

---

## Zbiór danych
Pełny zbiór CICIDS2017 (nie jest dołączony do repozytorium ze względu na rozmiar):  
➡ https://www.unb.ca/cic/datasets/ids-2017.html

Aby uruchomić analizę, należy pobrać pliki CSV z sekcji *Machine Learning Ready*.

---

## Ważna uwaga o danych
Zbiór CICIDS2017 jest **silnie niezbalansowany** – ruch BENIGN stanowi większość danych,
co wpływa na metryki oparte wyłącznie na accuracy.

W projekcie uwzględniono:
- analizę macierzy pomyłek,  
- wyniki dla poszczególnych klas,  
- wizualizacje rozkładu klas i wyników modeli.

Przy dalszym rozwijaniu projektu rekomendowane jest zastosowanie:
- F1-score (macro),  
- stratified sampling,  
- metod oversamplingu / undersamplingu.

---

## Wykorzystane modele

### Modele nadzorowane (supervised):
- **XGBoost**
- **Decision Tree Classifier**

### Modele nienadzorowane (unsupervised):
- **K-Means**
- **MiniBatch K-Means**
- **Gaussian Mixture Model (GMM)**
- **BIRCH**

---

## Etapy przetwarzania danych
- usunięcie braków danych  
- usunięcie obserwacji odstających (IQR)  
- skalowanie cech (StandardScaler)  
- podział danych na zbiór treningowy i testowy  
- próbkowanie danych z powodu ograniczeń pamięciowych (np. Google Colab)  
- użycie `random_state=42` dla zachowania powtarzalności wyników  

---

## Wyniki (podsumowanie)
- model **XGBoost** uzyskał najwyższą skuteczność (wysokie accuracy),  
- drzewo decyzyjne miało wyniki nieco słabsze, ale bardziej interpretowalne,  
- algorytmy klasteryzacji tworzyły wyraźne grupy obserwacji,  
- przeprowadzono analizę:  
  - macierzy pomyłek,  
  - ROC,  
  - ważności cech,  
  - silhouette score.  

Szczegółowe wyniki znajdują się w załączonej dokumentacji PDF.
