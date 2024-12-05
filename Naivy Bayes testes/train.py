# Load libraries
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB # importa a classe GaussianNB, que implementa o algoritmo Naive Bayes Gaussiano (Ele é comumente usado para tarefas de classificação, como o Iris)

# Load data
iris = datasets.load_iris() # 150 amostras de três espécies diferentes de flores do gênero Iris, e cada flor é descrita por quatro características (comprimento e largura da sépala e da pétala).
features = iris.data # uma matriz de valores numéricos com quatro colunas (as características das flores) e 150 linhas (Cada linha representa uma flor e suas características medidas.)
target = iris.target # extrai os rótulos ou classes (alvos) do dataset Iris (contém uma lista de inteiros, onde cada número representa a espécie da flor)

# Definindo as probabilidades a priori para as classes
priors = [0.25, 0.25, 0.5]  # Exemplo: Classe 0 tem 25%, Classe 1 tem 25%, Classe 2 tem 50%
classifer = GaussianNB(priors=priors) # cria uma instância do classificador Gaussian Naive Bayes
#train model
model = classifer.fit(features, target) #treina o classificador Naive Bayes utilizando os dados de características (features) e os rótulos (target). 
#O método fit() ajusta o modelo aos dados de treinamento, aprendendo a relação entre as características das flores (largura e comprimento das pétalas e sépalas) e suas respectivas espécies.

new_observation = [[4, 4, 4, 0.4]]
print(model.predict(new_observation))
