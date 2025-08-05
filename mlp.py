import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Definições para cálculo do WQI
wi = np.array([0.2213, 0.2604, 0.0022])
si = np.array([10, 8.5, 1000])
vIdeal = np.array([14.6, 7, 0])

def calc_wqi(sample):
    wqi_sample = 0
    num_col = 3
    for index in range(num_col):
        v_index = sample[index]
        v_index_ideal = vIdeal[index]
        w_index = wi[index]
        std_index = si[index]
        q_index = ((v_index - v_index_ideal) / (std_index - v_index_ideal)) * 100
        wqi_sample += q_index * w_index
    return wqi_sample

def process_data(file_path):
    df = pd.read_csv("water_dataX.csv", sep=',', encoding='unicode_escape')
    df = df.rename(columns={
        'D.O. (mg/l)': 'D.O.',
        'CONDUCTIVITY (µmhos/cm)': 'Conductivity',
        'B.O.D. (mg/l)': 'B.O.D',
        'NITRATENAN N+ NITRITENANN (mg/l)': 'NI',
        'FECAL COLIFORM (MPN/100ml)': 'Fecal_col',
        'TOTAL COLIFORM (MPN/100ml)Mean': 'Total_col',
    })
    df = df.drop(['STATION CODE', 'LOCATIONS', 'STATE', 'year', 'Temp'], axis=1)
    df.replace("NAN", np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    df['WQI'] = df.apply(lambda row: calc_wqi(row.values), axis=1)
    df = df[df['WQI'] >= 0]

    df['WQI clf'] = df['WQI'].apply(lambda x: 4 if x <= 25 else 3 if x <= 50 else 2 if x <= 75 else 1 if x <= 100 else 0)
    df = df[df['WQI'] <= 100]
    return df

def train_mlp(df):
    X = df.drop(columns=['WQI clf'])
    y = df['WQI clf']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)

    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=True,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000)


    kfold = StratifiedKFold(n_splits=5)
    cv_results = cross_validate(mlp, X, y, cv=kfold, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
    print(f"Precisão: {precision_score(y_test, y_pred, average='weighted',zero_division=1)}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted')}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=mlp.classes_, yticklabels=mlp.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    df = process_data('water_dataX.csv')
    train_mlp(df)
