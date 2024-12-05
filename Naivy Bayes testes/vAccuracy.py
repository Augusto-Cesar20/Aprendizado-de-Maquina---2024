from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# import scikit-learn metrics module for accuracy calculation
from sklearn import metrics # Esse módulo fornece funções para calcular métricas de avaliação de modelos, como a acurácia, que será usada para avaliar o desempenho do classificador.

from sklearn.model_selection import train_test_split # Essa função é usada para dividir um conjunto de dados em conjuntos de treinamento e teste.

iris = datasets.load_iris()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.3, random_state=109)
# X_train e y_train são os dados e rótulos usados para treinar o modelo.
# X_test e y_test são os dados e rótulos usados para testar o modelo.
# test_size=0.3 indica que 30% dos dados serão usados para teste e 70% para treinamento.
# random_state=109 garante que a divisão seja reproduzível, ou seja, a mesma divisão será obtida se o código for executado novamente com esse valor de random_state.

# Create a Gaussian Classifier
gnb = GaussianNB()

# Train the model using the train sets
gnb.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = gnb.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# A função accuracy_score() compara as previsões feitas pelo modelo (y_pred) com os rótulos reais (y_test) e calcula a proporção de previsões corretas.