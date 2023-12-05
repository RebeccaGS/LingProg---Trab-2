
# ---------------------------------------------------------------------------------------
# Trab 2 - LingProg - 23.2 - usar o aprendizado de máquina para criar um modelo que preveja quais
# passageiros sobreviveram ao naufrágio do Titanic.
# ---------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------
# 1- Importações Iniciais
# ---------------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# ---------------------------------------------------------------------------------------
# 2- leitura dos arquivos e separacao de colunas (loc/iloc)
# ---------------------------------------------------------------------------------------

dadosDeTreino = pd.read_csv('train.csv').iloc[:,:]
dadosDeTeste = pd.read_csv('test.csv').iloc[:,:]
respostasDeTeste =  pd.read_csv('gender_submission.csv').iloc[:,:]



# ---------------------------------------------------------------------------------------
# 3- Tratando os dados - Colunas: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# Eliminar colunas não tão úteis, tratar dados vazios e separar colunas com mais de duas respostas em categorias
# ---------------------------------------------------------------------------------------

# Iniciar tirando dados considerados não úteis
dadosDescartaveis = (["SibSp","Ticket","Parch","Fare","Cabin","Embarked","Name"])   
# descarte desses dados nas planilhas
dadosDeTreino = dadosDeTreino.drop(dadosDescartaveis,axis=1) # axis = 1, eliminar coluna 
dadosDeTeste = dadosDeTeste.drop(dadosDescartaveis,axis=1)


# Preenchendo vazios, transformando em not a number:
dadosDeTreino['Sex'] = np.where((dadosDeTreino.Sex == ' '),'N',dadosDeTreino.Sex)
dadosDeTeste['Sex'] = np.where((dadosDeTeste.Sex == ' '),'N',dadosDeTeste.Sex)

dadosDeTreino['Age'] = np.where((dadosDeTreino.Age == ' '),'XX',dadosDeTreino.Age)
dadosDeTeste['Age'] = np.where((dadosDeTeste.Age == ' '),'XX',dadosDeTeste.Age)


dadosDeTreino['Pclass'] = np.where((dadosDeTreino.Pclass == ' '),'XX',dadosDeTreino.Pclass)
dadosDeTeste['Pclass'] = np.where((dadosDeTeste.Pclass == ' '),'XX',dadosDeTeste.Pclass)

'''
# retirada de not a numbers - os not a number não estão sendo substituidos

dadosDeTreino['Age'] = dadosDeTreino['Age'].fillna(0)

categoriasComNAN = ['Sex','Age','Pclass']

for feature in categoriasComNAN:
    dadosDeTreino[feature] = dadosDeTreino[feature].fillna(0)
    dadosDeTeste[feature] = dadosDeTreino[feature].fillna(0)
    print('uia')
    
print('chegueiii')
print(dadosDeTeste)
print(dadosDeTreino)


for feature in categoriasComNAN:
    dadosDeTreino[feature] = dadosDeTreino[feature].replace(np.nan,0)
    dadosDeTeste[feature] = dadosDeTeste[feature].replace(np.nan,0)
    print('CHEGUEI')

'''

# Diferenciando nossas variáveis Dummies (Variáveis dummy são variáveis binárias criadas 
# para representar uma variável com duas ou mais categorias, sem hierarquia de relevância

colunasDummy = (['Sex',                            
                 'Age',
                 'Pclass'])

# Pegar Dummies
dadosDeTreino = pd.get_dummies(dadosDeTreino, columns = colunasDummy)
dadosDeTeste = pd.get_dummies(dadosDeTeste, columns = colunasDummy)

print(dadosDeTeste)
print(dadosDeTreino)

# ---------------------------------------------------------------------------------------
# 4 - Treinamento
# ---------------------------------------------------------------------------------------

# y_treinamento: pega so coluna de Survived \\ x_treinamento: pega todas as colunas menos as de Survived
y_treinamento = dadosDeTreino.loc[:,dadosDeTreino.columns == 'Survived'].values
x_treinamento = dadosDeTreino.loc[:,dadosDeTreino.columns != 'Survived'].values


# No treinamento, separar uma parte dos dados que sabemos as respostas pra estudar o padrão, e outra
# parte pra tentar prever e ver porcentagem de acerto
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split( # Func train test slip faz essa divisão
    x_treinamento, 
    y_treinamento.ravel(),
    train_size =    0.5, # usar 50% para treino e 10% para teste
)




# ---------------------------------------------------------------------------------------
# 5 - Testes
# ---------------------------------------------------------------------------------------