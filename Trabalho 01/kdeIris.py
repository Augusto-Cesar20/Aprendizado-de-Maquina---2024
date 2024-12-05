from sklearn import datasets
import math
from sklearn.model_selection import train_test_split # dividir um conjunto de dados em conjuntos de treinamento e teste.
from sklearn import metrics # fornece funções para calcular métricas de avaliação de modelos, como a acurácia, que será usada para avaliar o desempenho do classificador.

# Load data
iris = datasets.load_iris() # 150 amostras 
# quatro características (X): comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala
# classes (espécies) das flores [0 = Setosa, 1 = Versicolor e 2 = Virginica] (Y)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=109)

h = 0.5 # Bandwidth (Eu escolho testando)
d = 4

def arg(x, xi):

    return (math.sqrt( ((x[0] - xi[0])**2) + ((x[1] - xi[1])**2) + ((x[2] - xi[2])**2) + ((x[3] - xi[3])**2) )) / h

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

        sum_c0 = 0 # soma para classe 0 (Setosa)
        sum_c1 = 0 # soma para classe 1 (Versicolor)
        sum_c2 = 0 # soma para classe 2 (Virginica)

        N_c0 = 0 # Numero total de amostras da classe c0
        N_c1 = 0 # Numero total de amostras da classe c1
        N_c2 = 0 # Numero total de amostras da classe c2

        for i, line in enumerate(X_train):

            if(y_train[i] == 0):
                #print("Setosa")

                # Calculo do argumento u
                u = arg(test, line)
                # Calculo da soma da função kernel (Gaussiana)
                sum_c0 = sum_c0 + ((1 / math.sqrt(2 * math.pi))**d) * math.exp(-(u ** 2) / 2)

                N_c0 = N_c0 + 1

            elif(y_train[i] == 1):
                #print("Versicolor")

                # Calculo do argumento u
                u = arg(test, line)
                # Calculo da soma da função kernel (Gaussiana)
                sum_c1 = sum_c1 + ((1 / math.sqrt(2 * math.pi))**d) * math.exp(-(u ** 2) / 2)

                N_c1 = N_c1 + 1

            else:
                #print("Virginica")

                # Calculo do argumento u
                u = arg(test, line)
                # Calculo da soma da função kernel (Gaussiana)
                sum_c2 = sum_c2 + ((1 / math.sqrt(2 * math.pi))**d) * math.exp(-(u ** 2) / 2)

                N_c2 = N_c2 + 1

        N = N_c0 + N_c1 + N_c2

        # Janela de Parzen
        p_c0 = (1/N* (h**d)) * sum_c0

        p_c1 = (1/N* (h**d)) * sum_c1

        p_c2 = (1/N* (h**d)) * sum_c2

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

    # Gráfico do Erro (Acho q meu algoritimo não tem iterações para fazer um grafico de erro)
    print(y_test) #     [2 1 2 0 2 1 0 2 1 2 2 0 1 0 0 0 1 2 0 1 1 0 2 0 0 1 2 1 1 2 1 2 1 2 2 1 0 2 2 1 1 1 1 2 0] 
    print(predict) #    [2 1 1 0 2 1 0 2 1 2 2 0 1 0 0 0 1 1 0 1 2 0 2 0 0 1 2 1 1 2 1 2 1 2 1 1 0 2 2 1 1 1 1 2 0]

    erro_m = np.mean((y_test - predict) ** 2)
    print("Erro quadratico medio: ", erro_m)
    #print(metrics.mean_squared_error(y_test, predict))

    erro_r = np.sum(np.abs(y_test - predict)) / np.sum(np.abs(y_test - np.mean(y_test)))
    print("Erro quadratico relativo: ", erro_r)

###################################################################################################################################
    
    # Matriz de confusão
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    conf_matrix = confusion_matrix(y_test, predict)
    print(conf_matrix)
    # [[12  0  0]
    #  [ 0 16  1]
    #  [ 0  3 13]]

    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.show()