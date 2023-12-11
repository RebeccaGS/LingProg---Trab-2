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
from sklearn.metrics import accuracy_score


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
# print("dados de treino: \n"+str(dadosDeTreino.isnull().sum())+"\n") #deixamos comentadopois so foi usado para fins de consulta esporádicas 
# print("dados de teste: \n"+str(dadosDeTeste.isnull().sum()))

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

# Elimina colunas não tão úteis
dadosDescartaveis = ["Name", "Embarked"]
dadosDeTreino = dadosDeTreino.drop(dadosDescartaveis, axis=1)
dadosDeTeste = dadosDeTeste.drop(dadosDescartaveis, axis=1)


# ---------------------------------------------------------------------------------------
# 4 - Treinamento
# ---------------------------------------------------------------------------------------
# Seleciona as colunas importantes para previsão
colunas_utilizadas = ['Sex', 'Age', 'Cabin', 'Pclass', "SibSp", "Ticket", "Parch", "Fare"]

# Usar get_dummies para codificar variáveis categóricas
dadosDeTreino = pd.get_dummies(dadosDeTreino, columns=colunas_utilizadas, drop_first=True)
dadosDeTeste = pd.get_dummies(dadosDeTeste, columns=colunas_utilizadas, drop_first=True)

# Ajusta as colunas do conjunto de teste
dadosDeTeste = dadosDeTeste.reindex(columns=dadosDeTreino.columns, fill_value=0)

# Separa o conjunto de treinamento e teste
y_treinamento = dadosDeTreino['Survived']
x_treinamento = dadosDeTreino.drop('Survived', axis=1)

# Normaliza os dados
scaler = StandardScaler()
x_treinamento_scaled = scaler.fit_transform(x_treinamento)
x_predicao_scaled = scaler.transform(dadosDeTeste[x_treinamento.columns])

# Treina e preve usando RandomForestClassifier - os parâmetros foram testados e avaliados de acorod com a acurácia abaixo
modelo = RandomForestClassifier(n_estimators=10000, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=1)
modelo.fit(x_treinamento_scaled, y_treinamento)


# ---------------------------------------------------------------------------------------
# 5 - Previsõa
# ---------------------------------------------------------------------------------------
# Realiza as predições nos dados de teste
y_predicao = modelo.predict(x_predicao_scaled)

# Calcular a acurácia nos dados de treinamento
acuracia_treinamento = modelo.score(x_treinamento_scaled, y_treinamento)

# Cria o DataFrame com as colunas PassengerId e Survived assim como previsto na ementa do desafio
dadosPrevistos = pd.DataFrame({
    'PassengerId': dadosDeTeste['PassengerId'],
    'Survived': y_predicao
})

# Salva as predições em um arquivo CSV dentro da pasta de dados/
dadosPrevistos.to_csv('dados/predicoes.csv', index=False)

#Printa dados não necessários pelo desafio, porém interessantes para análise
print("Sobrevivencia: " + str(round(100*(dadosPrevistos['Survived'].sum()/dadosPrevistos['Survived'].count()))) + "%")
print("Quantidade total de passageiros: " + str(dadosPrevistos['Survived'].count()))
print("Acuracia: " + str(round(acuracia_treinamento*100, 2)) + "%")
