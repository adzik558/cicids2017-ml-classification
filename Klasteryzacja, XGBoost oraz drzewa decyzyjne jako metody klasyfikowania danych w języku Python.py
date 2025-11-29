import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
from xgboost import plot_importance

"""Wczytanie danych"""

df = pd.read_csv("combine.csv")
df_probka = df.sample(frac=0.3, random_state=42)

print(list(df_probka.columns))

(df_probka[' Label'].value_counts())

features = df_probka.dtypes[df_probka.dtypes != 'object'].index
df_probka[features] = df_probka[features].apply(
    lambda x: (x-x.mean())/(x.std()))

df_probka = df_probka.fillna(0)
dane_pre = df_probka.drop([' Label'], axis = 1)

print(dane_pre)

labelencoder = LabelEncoder()
df_probka.iloc[:, -1] = labelencoder.fit_transform(df_probka.iloc[:, -1])

dane_mniej = df_probka[(df_probka[' Label'] == 1 )|(df_probka[' Label'] == 8)|(df_probka[' Label'] == 7)]
dane_wiek = df_probka.drop(dane_mniej.index)
X = dane_wiek.drop([' Label'], axis = 1)
y = dane_wiek.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)
X = df_probka.drop([' Label'], axis=1).values
y = df_probka.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)

print(dane_wiek)

from sklearn.feature_selection import mutual_info_regression

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state = 0, stratify = y)

importances = mutual_info_regression(X_train, y_train)
f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
Sum = 0
fs = []

for i in range(0, len(f_list)):
    Sum = Sum + f_list[i][0]
    fs.append(f_list[i][1])
f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)

Sum2 = 0
fs = []

for i in range(0, len(f_list2)):
    Sum2 = Sum2 + f_list2[i][0]
    fs.append(f_list2[i][1])
    if Sum2 >= 0.9:
        break

X_fs = df_probka[fs].values
print(X_fs)

plt.plot(X_fs)
plt.xlabel("Numer obserwacji")
plt.ylabel("Wektor cechy")
plt.title("Wybór cechy")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_fs, y, train_size = 0.8, test_size=0.2, random_state=0, stratify = y)

# XGBoost Classifier
xg = xgb.XGBClassifier(n_estimators = 10)
xg.fit(X_train, y_train)

# Przewidywanie na zbiorze testowym
y_predict = xg.predict(X_test)

# Refit LabelEncoder with unique labels from y_train and y_test
labelencoder = LabelEncoder() # Reinitialize if you've previously used it
labelencoder.fit(np.unique(np.concatenate((y_train, y_test))))

# Zakodowanie etykiet do nazw
y_true = y_test
y_true_encoded = labelencoder.transform(y_true)
y_predict_encoded = labelencoder.transform(y_predict)

# Obliczanie metryk
xg_score = xg.score(X_test, y_true_encoded)
print('Accuracy of XGBoost: ' + str(xg_score))

precision, recall, fscore, none = precision_recall_fscore_support(y_true_encoded, y_predict_encoded, average='weighted')
print('Precision of XGBoost: ' + str(precision))
print('Recall of XGBoost: ' + str(recall))
print('F1-score of XGBoost: ' + str(fscore))

print(classification_report(y_true_encoded, y_predict_encoded))

# Macierz pomyłek
cm = confusion_matrix(y_true_encoded, y_predict_encoded)
f, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(cm, annot = True, linewidths = 0.5, linecolor = "blue", fmt = ".0f", ax = ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state = 0)
# Ensure y_train has the correct type and values
y_train_encoded = labelencoder.transform(y_train) # Encode y_train using the same LabelEncoder
dt.fit(X_train, y_train_encoded) # Fit with encoded y_train

# Przewidywanie na zbiorze testowym
y_predict = dt.predict(X_test)

# Zakodowanie etykiet do nazw
y_true_encoded = labelencoder.transform(y_true)
y_predict_encoded = labelencoder.transform(y_predict)
# Obliczanie metryk
dt_score = dt.score(X_test, y_true_encoded)
print('Accuracy of DT: ' + str(dt_score))

precision, recall, fscore, none = precision_recall_fscore_support(y_true_encoded, y_predict_encoded, average='weighted')
print('Precision of DT: ' + str(precision))
print('Recall of DT: ' + str(recall))
print('F1-score of DT: ' + str(fscore))

print(classification_report(y_true_encoded, y_predict_encoded))

# Macierz pomyłek dla drzewa decyzyjnego
cm = confusion_matrix(y_true_encoded, y_predict_encoded)
f, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(cm, annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f", ax = ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

"""Histogram rozkładu klas"""

# Histogram rozkładu klas
plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title("Rozkład klas w zbiorze danych")
plt.xlabel("Klasa")
plt.ylabel("Liczba próbek")
plt.show()

# Wykres ważności cech
plt.figure(figsize=(20, 15))
importance_plot = plot_importance(xg, importance_type="gain", show_values=False)

# Dodanie tytułu i układu
plt.title("Ważność cech (XGBoost)")
plt.tight_layout()
plt.show()

# Wykres rozkładu predykcji dla klasy "Normal" (klasa 0)
plt.figure(figsize=(8, 6))
sns.distplot(xg.predict_proba(X_test)[:, 0], color="blue", label="Normal")
plt.title("Rozkład predykcji dla klasy Normal")
plt.xlabel("Prawdopodobieństwo")
plt.ylabel("Gęstość")
plt.legend()
plt.show()

from sklearn.metrics import roc_curve, auc
# Krzywa ROC dla klasy "Benign" (klasa 0)
fpr, tpr, thresholds = roc_curve(y_true_encoded, xg.predict_proba(X_test)[:, 0], pos_label=0)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()

"""#Klasteryzacja

Usuwanie braków w danych
"""

df = df.dropna()

"""Sprawdzamy rozmiar ramki danych"""

df.shape

"""Wypisujemy nazwy kolumn"""

column_names = df.columns
print(column_names)

"""Podgląd unikalnych wartości w kolumnie Label, w której wypisany został typ ataku sieciowego"""

unique_values = df[' Label'].unique()
print(unique_values)

"""Tworzymy odpowiedniki liczbowe każdego z typów ataków"""

label_clusters = {
    'BENIGN': 0,
    'Bot': 1,
    'DDoS': 2,
    'DoS GoldenEye': 3,
    'DoS Hulk': 4,
    'DoS Slowhttptest': 5,
    'DoS slowloris': 6,
    'Heartbleed': 7,
    'Infiltration': 8,
    'PortScan': 9
}

"""#Przygotowanie danych

Nadpisujemy wartości w kolumnie Label na wartości numeryczne
"""

df[' Label'] = df[' Label'].map(label_clusters).astype('category')

import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='Reds')
plt.title('Correlation Matrix')
plt.show()

"""Usuwanie wartości odstających metodą Interquartile Range (rozstęp międzykwartylowy)"""

def us_wart_odst(df, multiplier):
    df_numeric = df.select_dtypes(include=[float, int])
    q05 = df_numeric.quantile(0.05)
    q95 = df_numeric.quantile(0.95)

    IQR = q95 - q05

    nieodstaje = ~((df_numeric < (q05 - multiplier * IQR)) | (df_numeric > (q95 + multiplier * IQR))).any(axis=1)
    return df[nieodstaje]

df_iqr = us_wart_odst(df, 10).dropna()

"""Sprawdzamy rozmiar ramki danych po przeprowadzeniu metody IQR"""

df_iqr.shape

"""Danych jest bardzo dużo, dlatego weźmiemy ich próbkę"""

df_probka = df_iqr.sample(frac=0.5, random_state=42)

(df_probka[' Label'].value_counts())

unique_values = df_probka[' Label'].unique()
print(unique_values)

"""Sprawdzamy rozmiar ramki danych po wyciągnięciu z niej próbki"""

df_probka.shape

"""Standaryzujemy dane"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_stanaryz = scaler.fit_transform(df_probka.drop(' Label', axis=1))

"""Redukujemy liczbę kolumn metodą PCA (Analiza głównych składowych)"""

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_stanaryz)

"""Tworzymy ramkę danych z dwóch głównych składowych, które powstały po przeprowadzeniu PCA"""

df_glowne = pd.DataFrame(data = df_pca,
                         columns = ['Główna składowa 1', 'Główna składowa 2'])

df_glowne

"""Dodajemy jeszcze kolumnę z określonym atakiem występującym dla danych z wierszy"""

df_glowne = df_glowne.reset_index(drop=True)
df_probka = df_probka.reset_index(drop=True)

df_glowne_atak = df_glowne

df_glowne_atak['Typ ataku'] = df_probka[' Label']
df_glowne_atak

unique_values = df_glowne_atak['Typ ataku'].unique()
print(unique_values)

"""#Klasteryzacja

Importujemy biblioteki do generowania wykresów klasteryzacji
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""Klasteryzacja metodą MiniBatchKMeans"""

from sklearn.cluster import MiniBatchKMeans

# Wyznaczamy liczbę klastrów
mbk = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=1000)
mbk_labels = mbk.fit_predict(df_pca)

# Do stworzenia legendy użyjemy odwróconą wersję zmiennej label_clusters z początku skryptu
cluster_labels = {
    0: 'BENIGN',
    1: 'Bot',
    2: 'DDoS',
    3: 'DoS GoldenEye',
    4: 'DoS Hulk',
    5: 'DoS Slowhttptest',
    6: 'DoS slowloris',
    7: 'Heartbleed',
    8: 'Infiltration',
    9: 'PortScan'
}

# Wizualizacja wyników klasteryzacji metodą MiniBatchKMeans
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=mbk_labels, cmap='tab10', s=1, alpha=1)
plt.title("Klasteryzacja metodą MiniBatchKMeans")
plt.xlabel("Główna składowa 1")
plt.ylabel("Główna składowa 2")
handles, labels = scatter.legend_elements()
new_labels_mbk = [cluster_labels[int(label)] for label in np.unique(mbk_labels)]
plt.legend(handles, new_labels_mbk, title="Typ ataku", loc='upper right')
plt.show()

"""Klasteryzacja metodą KMeans"""

from sklearn.cluster import KMeans

# Wyznaczamy liczbę klastrów
metodakmeans = KMeans(n_clusters=10, random_state=42)
kmeans_labels = metodakmeans.fit_predict(df_pca)

# Wizualizacja wyników klasteryzacji metodą K-Means
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels, cmap='tab10', s=1, alpha=1)
plt.title("Klasteryzacja metodą KMeans")
plt.xlabel("Główna składowa 1")
plt.ylabel("Główna składowa 2")
new_labels_km = [cluster_labels[int(label)] for label in np.unique(kmeans_labels)]
plt.legend(handles, new_labels_km, title="Typ ataku", loc='upper right')
plt.show()

"""Klasteryzacja metodą Gaussian Mixtures Model"""

from sklearn.mixture import GaussianMixture

# Wyznaczamy liczbę klastrów
metodagmm = GaussianMixture(n_components=10, random_state=42)
gmm_labels = metodagmm.fit_predict(df_pca)

# Wizualizacja wyników klasteryzacji metodą GMM
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=gmm_labels, cmap='tab10', s=1, alpha=1)
plt.title("Klasteryzacja metodą Gaussian Mixture Model ")
plt.xlabel("Główna składowa 1")
plt.ylabel("Główna składowa 2")
new_labels_GMM = [cluster_labels[int(label)] for label in np.unique(gmm_labels)]
plt.legend(handles, new_labels_GMM, title="Attack Type", loc='upper right')
plt.show()

"""Klasteryzacja metodą BIRCH"""

from sklearn.cluster import Birch

# Określamy liczbę klastrów
metodabirch = Birch(n_clusters=10)
birch_labels = metodabirch.fit_predict(df_pca)

# Wizualizacja wyników klasteryzacji metodą Birch
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=birch_labels, cmap='tab10', s=2, alpha=1)
plt.title("Klasteryzacja metodą BIRCH")
plt.xlabel("Główna składowa 1")
plt.ylabel("Główna składowa 2")
new_labels_BIRCH = [cluster_labels[int(label)] for label in np.unique(birch_labels)]
plt.legend(handles, new_labels_BIRCH, title="Typ ataku", loc='upper right')
plt.show()

"""#Sprawdzanie wyników

Obliczanie poprawności klasteryzacji przy pomocy funkcji Silhouette_score
"""

from sklearn.metrics import silhouette_score

mbk_silhouette = silhouette_score(df_pca, mbk_labels)
print(f"MiniBatchKMeans Silhouette Score: {mbk_silhouette}")

kmeans_silhouette = silhouette_score(df_pca, kmeans_labels)
print(f"KMeans Silhouette Score: {kmeans_silhouette}")

gmm_silhouette = silhouette_score(df_pca, gmm_labels)
print(f"GMM Silhouette Score: {gmm_silhouette}")

birch_silhouette = silhouette_score(df_pca, birch_labels)
print(f"Birch Silhouette Score: {birch_silhouette}")

"""Drzewo decyzyjne"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree

df_glowne_atak['Typ ataku'] = df_glowne_atak['Typ ataku'].astype('category')

X = df_glowne_atak[['Główna składowa 1', 'Główna składowa 2']]
y = df_glowne_atak['Typ ataku']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Trenowanie drzewa
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=10)
dt_classifier.fit(X_train, y_train)
# Predykcja
y_pred = dt_classifier.predict(X_test)

# Raport dokładności
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Tworzenie listy nazw ataków na podstawie mapowania
class_labels = [cluster_labels[i] for i in sorted(cluster_labels.keys())]

# Wizualizacja drzewa decyzyjnego
plt.figure(figsize=(30, 10))
plot_tree(
    dt_classifier,
    feature_names=['Główna składowa 1', 'Główna składowa 2'],
    class_names=class_labels,
    filled=True,
    fontsize=10
)
plt.title(" Drzewo decyzyjne ataków sieciowych")
plt.show()