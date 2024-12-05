from sklearn import datasets
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # dividir um conjunto de dados em conjuntos de treinamento e teste.
from sklearn import metrics # fornece funções para calcular métricas de avaliação de modelos, como a acurácia, que será usada para avaliar o desempenho do classificador.

# Load data
iris = datasets.load_iris() # 150 amostras 
# quatro características (X): comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala
# classes (espécies) das flores [0 = Setosa, 1 = Versicolor e 2 = Virginica] (Y)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=109)

k = 11 # Eu escolho (geralmente ímpar)

if __name__ == "__main__":

    predict = []

    print("###################################################")

    print(len(X_train))
    print(X_train)
    print(len(y_train))
    print(y_train)
    
    print("###################################################")

    print(len(X_test))
    print(X_test)
    print(len(y_test))
    print(y_test)

    print("###################################################")

    for j, test in enumerate(X_test):
        
        dist = []

        # Descobrir a distancia entre minha observação de teste e os valores de treinamento
        for i, line in enumerate(X_train):
            dist.append(math.sqrt( ((test[0] - line[0])**2) + ((test[1] - line[1])**2) + ((test[2] - line[2])**2) + ((test[3] - line[3])**2) ))


        min = []
        index = []

        ki_c0 = 0
        ki_c1 = 0
        ki_c2 = 0

        # Descobrir os que tem a menor distancia em relação a minha observação e somar a quantidade de vezes que cada uma das classes aparece
        for j in range(k):
            min.append(sorted(dist)[j])
            index.append(dist.index(min[j]))

            if (y_train[index[j]] == 0):
                ki_c0 = ki_c0 + 1
            elif (y_train[index[j]] == 1):
                ki_c1 = ki_c1 + 1
            else:
                ki_c2 = ki_c2 + 1

        p_c0 = ki_c0/k
        p_c1 = ki_c1/k
        p_c2 = ki_c2/k

        if (p_c0 > p_c1 and p_c0 > p_c2):
            predict.append(0)
        elif (p_c1 > p_c0 and p_c1 > p_c2):
            predict.append(1)
        else:
            predict.append(2)


    print(len(predict))
    print(predict)

    print("###################################################")

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, predict))
    print("Precision:", metrics.precision_score(y_test, predict, average='macro'))
    print("Recall:", metrics.recall_score(y_test, predict, average='macro'))
    print("F1-Score:", metrics.classification_report(y_test, predict))

###################################################################################################################################
    #GRAFICOS

    import numpy as np
    import matplotlib.pyplot as plt

    predict = np.array(predict)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    feature_x_index = 0  # comprimento da sépala
    feature_y_index = 1  # largura da sépala

    # Criando subplots para comparar dados de treino e teste
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plotando os dados de treino
    ax[0].scatter(X_train[y_train == 0, feature_x_index], X_train[y_train == 0, feature_y_index], color='r', label='Setosa (train)')
    ax[0].scatter(X_train[y_train == 1, feature_x_index], X_train[y_train == 1, feature_y_index], color='g', label='Versicolor (train)')
    ax[0].scatter(X_train[y_train == 2, feature_x_index], X_train[y_train == 2, feature_y_index], color='b', label='Virginica (train)')
    ax[0].set_title('Train Data')
    ax[0].set_xlabel('Sepal Length')
    ax[0].set_ylabel('Sepal Width')
    ax[0].legend()

    # Plotando os dados de teste com a previsão
    ax[1].scatter(X_test[predict == 0, feature_x_index], X_test[predict == 0, feature_y_index], color='r', label='Setosa (pred)')
    ax[1].scatter(X_test[predict == 1, feature_x_index], X_test[predict == 1, feature_y_index], color='g', label='Versicolor (pred)')
    ax[1].scatter(X_test[predict == 2, feature_x_index], X_test[predict == 2, feature_y_index], color='b', label='Virginica (pred)')
    ax[1].set_title('Test Data Predictions')
    ax[1].set_xlabel('Sepal Length')
    ax[1].set_ylabel('Sepal Width')
    ax[1].legend()

    plt.show()

#####################################################################

    # Gráfico do Erro 
    print(y_test) #    
    print(predict) #    

    erro_m = np.mean((y_test - predict) ** 2)
    print(erro_m)
    #print(metrics.mean_squared_error(y_test, predict))

    erro_r = np.sum(np.abs(y_test - predict)) / np.sum(np.abs(y_test - np.mean(y_test)))
    print(erro_r)

###################################################################################################################################

    # Matriz de confusão
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    conf_matrix = confusion_matrix(y_test, predict)
    print(conf_matrix)
    # [[12  0  0]
    #  [ 0 16  1]
    #  [ 0  1 15]]

    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.show()