from sklearn import datasets # fornecer conjuntos de dados prontos para uso
iris = datasets.load_iris() # dados brutos, informações sobre 150 amostras de flores da espécie Iris (divididas em 3 classes diferentes: Setosa, Versicolor e Virginica).

# As amostras são descritas por quatro características (ou features): comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala, 
# e o objetivo é prever a classe da flor (espécie) com base nessas características.

print(iris.data) # Ver as características das flores (dados brutos)
print(iris.feature_names) # Imprimir os nomes das features (características)
print(iris.target) # Ver as classes (espécies) das flores [0 = Setosa, 1 = Versicolor e 2 = Virginica]