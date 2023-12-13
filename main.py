# ---------------------------------------------------------------------------------------
# Trab 2 - LingProg - 23.2 - usar o aprendizado de máquina para criar um modelo que preveja quais
# passageiros sobreviveram ao naufrágio do Titanic.
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# 1- Importações Iniciais
# ---------------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
import random
import numpy as np


# ---------------------------------------------------------------------------------------
# 2- leitura dos arquivos para manipulação
# ---------------------------------------------------------------------------------------
# Carrega os dados de treino e de teste
dadosDeTreino = pd.read_csv('dados/train.csv')
dadosDeTeste = pd.read_csv('dados/test.csv')


# ---------------------------------------------------------------------------------------
# 3- Tratando os dados - Colunas: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# Eliminar colunas não tão úteis, tratar dados vazios e separar colunas com mais de duas respostas em categorias
# ---------------------------------------------------------------------------------------
# Os outros dados não precisam de tratamento pois não possuem dados vazios, como obervado no código abaixo
# print("dados de treino: \n"+str(dadosDeTreino.isnull().sum())+"\n") #deixamos comentado pois so foi usado para fins de consulta esporádicas 
# print("dados de teste: \n"+str(dadosDeTeste.isnull().sum()))
# print("\n"+str(dadosDeTreino['Fare'].unique)+"\n")

# Função para realizar o tratamento de idades vazias de acordo com o pronome utilizado
def tratamento_de_idade(row):
    """A função recebe uma linha do DataFrame e retorna a média de idade de acordo com o pronome de tratamento
    para se ter um tratamento de dados faltosos mais adequado."""
    #Verifica se a idade é nula e se o pronome de tratamento analisado está presente no nome
    if pd.isnull(row['Age']) and 'Mr.' in row['Name']:
        #retorna o arredondamento da média das idades que possuem o pronome de tratamento utilizado
        return round(dadosDeTreino[dadosDeTreino['Name'].str.contains('Mr.')]['Age'].mean())
    elif pd.isnull(row['Age']) and 'Miss.' in row['Name']:
        return round(dadosDeTreino[dadosDeTreino['Name'].str.contains('Miss.')]['Age'].mean())
    elif pd.isnull(row['Age']) and 'Mrs.' in row['Name']:
        return round(dadosDeTreino[dadosDeTreino['Name'].str.contains('Mrs.')]['Age'].mean())
    elif pd.isnull(row['Age']) and 'Dr.' in row['Name']:
        return round(dadosDeTreino[dadosDeTreino['Name'].str.contains('Dr.')]['Age'].mean())
    elif pd.isnull(row['Age']) and 'Master.' in row['Name']:
        return round(dadosDeTreino[dadosDeTreino['Name'].str.contains('Master.')]['Age'].mean())
    elif pd.isnull(row['Age']) and 'Ms.' in row['Name']:
        return round(dadosDeTreino[dadosDeTreino['Name'].str.contains('Ms.')]['Age'].mean())
    else:
        return row['Age']

# Aplica a função de tratamento acima usando apply 
dadosDeTreino['Age'] = dadosDeTreino.apply(lambda row: tratamento_de_idade(row), axis=1)
dadosDeTeste['Age'] = dadosDeTeste.apply(lambda row: tratamento_de_idade(row), axis=1)

# Trata o gênero como binário - não era necessário pois não haviam dados faltosos, porém foi feito pois é uma boa prática
dadosDeTreino.loc[dadosDeTreino['Sex'] == 'male', ['Sex']] = 0
dadosDeTreino.loc[dadosDeTreino['Sex'] == 'female', ['Sex']] = 1
dadosDeTeste.loc[dadosDeTeste['Sex'] == 'male', ['Sex']] = 0
dadosDeTeste.loc[dadosDeTeste['Sex'] == 'female', ['Sex']] = 1

# Cria uma nova coluna representando o tamanho total da família a bordo
dadosDeTreino['TamnhoDaFamilia'] = dadosDeTreino['SibSp'] + dadosDeTreino['Parch'] + 1
dadosDeTeste['TamnhoDaFamilia'] = dadosDeTeste['SibSp'] + dadosDeTeste['Parch'] + 1

# Elimina colunas não tão úteis
dadosDescartaveis = ["Name", "Embarked", "Ticket", "SibSp", "Parch", "Cabin"]
dadosDeTreino = dadosDeTreino.drop(dadosDescartaveis, axis=1)
dadosDeTeste = dadosDeTeste.drop(dadosDescartaveis, axis=1)

# Seleciona as colunas importantes para previsão
colunas_utilizadas = ["Sex", "Age", "Pclass", "TamnhoDaFamilia", "Fare"]

# Usar get_dummies para codificar variáveis categóricas
dadosDeTreino = pd.get_dummies(dadosDeTreino, columns=colunas_utilizadas, drop_first=True)
dadosDeTeste = pd.get_dummies(dadosDeTeste, columns=colunas_utilizadas, drop_first=True)

# Ajusta as colunas do conjunto de teste
dadosDeTeste = dadosDeTeste.reindex(columns=dadosDeTreino.columns, fill_value=0)


# ---------------------------------------------------------------------------------------
# 4 - Treinamento e Previsõa
# ---------------------------------------------------------------------------------------
# Separa o conjunto de treinamento e teste
y_treinamento = dadosDeTreino['Survived']
x_treinamento = dadosDeTreino.drop('Survived', axis=1)

# Normaliza os dados
scaler = StandardScaler()
x_treinamento_scaled = scaler.fit_transform(x_treinamento)
x_predicao_scaled = scaler.transform(dadosDeTeste[x_treinamento.columns])

# Número de execuções para realizar a previsão
numero_de_execucoes = 1000
numero_de_cenarios = 0
# Inicializa um DataFrame vazio para armazenar a soma
dadosPrevistos = None

# Alógica do for abaixo serve para garantir a estabilidade da previsão e tentar melhorá-la por conseuqnete 
for i in range(numero_de_execucoes):
    x_treino, x_validacao, y_treino, y_validacao = train_test_split(x_treinamento_scaled, y_treinamento, test_size=0.5, random_state=random.randint(1, 10**9))

    # Usa o RandomForestClassifier para realizar o treinamento
    modelo = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=random.randint(1, 10**9))
    modelo.fit(x_treino, y_treino)

    acuracia = np.mean(modelo.predict(x_validacao) == y_validacao)

    if acuracia >= 0.82:
        # Adiciona as predições ao DataFrame de previsão
        numero_de_cenarios += 1
        if dadosPrevistos is None:
            dadosPrevistos = pd.DataFrame({
                'PassengerId': dadosDeTeste['PassengerId'],
                'Survived': modelo.predict(x_predicao_scaled)
            })
        else:
            dadosPrevistos['Survived'] += modelo.predict(x_predicao_scaled)*acuracia

if dadosPrevistos is not None and numero_de_cenarios > 0:
    # Divide as predições pela quantidade de execuções e arredonda para o inteiro mais próximo - garantindo assim estar em binário
    dadosPrevistos['Survived'] = round(dadosPrevistos['Survived'] / numero_de_cenarios).astype(int)
    # Salva o resultado da soma em um novo arquivo .CSV
    dadosPrevistos.to_csv('dados/predicoes.csv', index=False)

    #Printa dados não necessários pelo desafio, porém interessantes para análise
    print("Sobrevivencia: " + str(round(100*(dadosPrevistos['Survived'].sum()/dadosPrevistos['Survived'].count()))) + "%")
    print("Quantidade total de passageiros: " + str(dadosPrevistos['Survived'].count()))
    print("Quantidade de sobreviventes: " + str(dadosPrevistos['Survived'].sum()))
    print("Quantidade de mortos: " + str(dadosPrevistos['Survived'].count()-dadosPrevistos['Survived'].sum()))
    print(str(numero_de_cenarios) + " execuções")

else:
    print("Não foi possível realizar a previsão com a acurácia desejada.")
