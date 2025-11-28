#Importação das bibliotecas
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import joblib

#Trazendo o dataset
df = pd.read_csv("dataSet/dataset.csv", sep=";")
dfEx = pd.read_csv("dataSet/dataset2.csv", sep=";")
casesCorrigidos = pd.read_csv("dataSet/casesCorrigidos.csv", sep=";")
dfFinal = pd.concat([df,dfEx,casesCorrigidos], ignore_index=True)
print(dfFinal.head())


#Tirando a coluna risco para treino
x = dfFinal.drop("risco", axis=1)
y = dfFinal["risco"]


#Pré-processamento 
preprocessador = ColumnTransformer(
    transformers=[
        ("texto", TfidfVectorizer(), "sintomas"),
        ("categorico", OneHotEncoder(), ["doenca_cronica", "consegue_mover"]),
        ("numerico", "passthrough", ["idade", "temperatura", "tempo_sintomas_h", "nivel_dor"])
    ]
)

modelo = Pipeline([
    ("preprocessamento", preprocessador),
    ("classificador", RandomForestClassifier())
])


#Treino

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3
)
modelo.fit(x_train, y_train)

joblib.dump(modelo, "modelo.pkl")
print("Modelo salvo!")

'''
#Avaliação

pred = modelo.predict(x_test)
acuracia = (accuracy_score(y_test,pred))
print("---- Acurácia ----")
print(acuracia)


#Função para entrada de dados
def classificarPaciente():
    print("---- Sistema de Triagem Automática ----")
    
    sintomas = input("Descreva seus sintomas: ")
    idade = int(input("Idade: "))
    temperatura = float(input("Temperatura corporal: "))
    doenca_cronica = input("Possui alguma doença crônica (Asma, Diabetes ou hipertensão) caso não tenha, digite: 'nenhuma': ")
    tempo_sintomas = float(input("Tempo de sintomas em horas: "))
    nivel_dor = int(input("Nível de dor (0-10): "))
    consegue_mover = input("Consegue se mover sozinho? (sim/não): ")
    
    dados_usuario = pd.DataFrame({
        "sintomas": [sintomas],
        "idade": [idade],
        "temperatura": [temperatura],
        "doenca_cronica": [doenca_cronica],
        "tempo_sintomas_h": [tempo_sintomas],
        "nivel_dor": [nivel_dor],
        "consegue_mover": [consegue_mover]
    })
    
    resultado = modelo.predict(dados_usuario)
    
    print("---- Resultado da Triagem ----")
    print("Classificação de risco:", resultado)
    
classificarPaciente()


#Distribuição das Classes de Risco#
df["risco"].value_counts().plot(kind="bar")
plt.title("Distribuição das Classes de Risco")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

#Matriz de Confusão#
y_pred = modelo.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Matriz de Confusão do Modelo de Triagem")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

#Acurácia#
plt.bar(["Modelo de triagem"],[acuracia])
plt.ylim(0,1)
plt.title("Acurácia do Modelo de Triagem")
plt.ylabel('Acurácia')
plt.show()

rf = modelo.named_steps["classificador"]
importances = rf.feature_importances_
nomes = modelo.named_steps["preprocessamento"].get_feature_names_out()
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), nomes[indices], rotation=90)
plt.title("Importância das Variáveis no Modelo")
plt.tight_layout()
plt.show()
'''