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
# Função para realizar o tratamento de idades vazias de acordo com o pronome utilizado
def tratamento_de_idade(row):
    if pd.isnull(row['Age']) and 'Mr.' in row['Name']:
        return dadosDeTreino[dadosDeTreino['Name'].str.contains('Mr.')]['Age'].mean()

    elif pd.isnull(row['Age']) and 'Miss.' in row['Name']:
        return dadosDeTreino[dadosDeTreino['Name'].str.contains('Miss.')]['Age'].mean()
    
    elif pd.isnull(row['Age']) and 'Mrs.' in row['Name']:
        return dadosDeTreino[dadosDeTreino['Name'].str.contains('Mrs.')]['Age'].mean()
    
    elif pd.isnull(row['Age']) and 'Dr.' in row['Name']:
        return dadosDeTreino[dadosDeTreino['Name'].str.contains('Dr.')]['Age'].mean()
    
    elif pd.isnull(row['Age']) and 'Master.' in row['Name']:
        return dadosDeTreino[dadosDeTreino['Name'].str.contains('Master.')]['Age'].mean()

# Aplica a função de tratamento acima usando apply 
dadosDeTreino['Age'] = dadosDeTreino.apply(lambda row: tratamento_de_idade(row), axis=1)
dadosDeTeste['Age'] = dadosDeTreino.apply(lambda row: tratamento_de_idade(row), axis=1)

# Tratamento de dados vazios em Sex - Tentamos aplicar um método similar ao da idade, porém a acurácia diminuiu ao inves de aumentar
dadosDeTreino['Sex'] = dadosDeTreino['Sex'].replace('', 'N')
dadosDeTeste['Sex'] = dadosDeTeste['Sex'].replace('', 'N')

# Tratamento de dados vazios em Pclass
dadosDeTreino['Pclass'] = dadosDeTreino['Pclass'].replace('', 'XX')
dadosDeTeste['Pclass'] = dadosDeTeste['Pclass'].replace('', 'XX')

# Tratamento de dados vazios em Embarked
dadosDeTreino['Embarked'] = dadosDeTreino['Embarked'].replace('', 'XX')
dadosDeTeste['Embarked'] = dadosDeTeste['Embarked'].replace('', 'XX')

# Eliminar colunas não tão úteis
dadosDescartaveis = ["SibSp", "Ticket", "Parch", "Fare", "Name"]
dadosDeTreino = dadosDeTreino.drop(dadosDescartaveis, axis=1)
dadosDeTeste = dadosDeTeste.drop(dadosDescartaveis, axis=1)


# ---------------------------------------------------------------------------------------
# 4 - Treinamento
# ---------------------------------------------------------------------------------------
# Seleciona as colunas importantes para previsão
colunas_utilizadas = ['Sex', 'Age', 'Cabin', 'Embarked', 'Pclass']

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

# Treina e preve usando RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(x_treinamento_scaled, y_treinamento)
y_predicao = modelo.predict(x_predicao_scaled)


# ---------------------------------------------------------------------------------------
# 5 - Previsõa
# ---------------------------------------------------------------------------------------
# Realiza as predições nos dados de teste
y_predicao = modelo.predict(x_predicao_scaled)

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
