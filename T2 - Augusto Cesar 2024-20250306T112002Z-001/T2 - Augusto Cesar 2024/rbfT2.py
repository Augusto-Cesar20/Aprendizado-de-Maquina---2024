from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split # dividir um conjunto de dados em conjuntos de treinamento e teste.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Gerar um dataset 
X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=109)

# Normalizar X e y
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

# Neuronios da camada oculta (H)
H = 12

# Centros (m):
m = [
    [0.69321602, -0.83645056, 0.46356806, 1.0588505],
    [0.83322947, 0.6626997, -0.14475505, 0.15257985],
    [-1.16554823, 0.20852695, -0.0820158, -0.5065535],
    [-1.03920593, 0.48854546, 0.75655635, -2.54265183],
    [-1.02496614, -0.85763018, -0.45610099, 0.33885959],
    [1.60826194, -0.42876896, 1.21270681, -0.01765857],
    [0.87782818, 0.0217352, 1.44171054, 0.23357005],
    [-0.70060563, 0.64368139, 1.24567399, -0.70215599],
    [-0.30198368, -0.80411294, 1.40516094, 0.38811918],
    [1.66519542, -1.00305709, -1.33229093, 0.81620114],
    [-0.59933721, -0.88871029, 2.70523491, -1.03686115],
    [0.22861962, -0.87257873, 0.25379815, -0.09882513]
]
#m = X[np.random.choice(len(X), H, replace=False)]
#print("m=>", m)

# Estpalhamento (sh):
s = 0.5

# Pesos (w): 
w = [-1.59088316, 0.5591758, 1.12198116, 0.62340975, 0.18093274, -0.70016163,
      0.83453995, 1.36510077, 0.66412175, 0.89119888, 0.61414482, 0.61751515]
#w = np.random.uniform(-2, 2, H)
#print("w=>", w)

# Bias (w0)
bias = 0.01

# Taxa de aprendizado (n)
n = 0.1 

# Erro em batelada
erro_b = []


#### TREINAMENTO
for epoca in range(1000):
    predic = [] # Saida dos meus neuronios
    erro = [] # Erro local
    for linha in range(len(X_train)):
        ## GAUSSIANA
        p = []
        for h in range(H):
            p.append( np.exp(-np.linalg.norm(X_train[linha] - m[h])**2 / (2 * s**2)) )

        ## PREVISÃO
        soma_yp = 0
        for h in range(H):
            soma_yp = soma_yp + p[h] * w[h]
        predic.append(soma_yp + bias)

        ## ERRO
        erro.append(y_train[linha] - predic[linha])

        ## Atualizações de PESO, CENTRO e ESPALHAMENTO
        novo_w =[]
        novo_m = []
        novo_s = 0
        soma_s = 0
        novo_bias = 0
        for h in range(H):
            ## Novos Pesos
            novo_w.append( w[h] + n * erro[linha] * p[h] )

            ## Novos Centros
            novo_m.append( m[h] + n * erro[linha] * p[h] * ( ((X_train[linha] - m[h]) / s**2) ) )

            ## novo Espalahmento
            soma_s = soma_s + n * erro[linha] * w[h] * p[h] * (np.linalg.norm(X_train[linha] - m[h])**2) / s**3
        novo_s = s + soma_s

        novo_bias = bias + n * erro[linha]

        ## Troca dos Valores
        w = novo_w
        bias = novo_bias
        m = novo_m
        s = novo_s 

    ## Erro em Batelada
    soma_erro = 0
    for i in range(len(X_train)):
        soma_erro = soma_erro + (erro[i]**2)
    erro_b.append( soma_erro/2 )

#print("Erro em batelada", erro_b)
print("##################################")
print("Erro em batelada", erro_b[-1])
#print("Pesos (w):", w)
#print("Bias (w0):", bias)
#for h in range(H):
    #print("Centro m(", h+1, "): ", m[h])
#print("Espalhamento (s)", s)
print("##################################")


#### TESTE
y_pred = []
for linha in range(len(X_test)):
    ## GAUSSIANA
    p = []
    for h in range(H):
        p.append( np.exp(-np.linalg.norm(X_test[linha] - m[h])**2 / (2 * s**2)) )

    ### PREVISÃO
    soma_yp = 0
    for h in range(H):
        soma_yp = soma_yp + p[h] * w[h]
    y_pred.append(soma_yp + bias)
    print("<<<<<(", linha, ")>>>>>")
    print("y_pred=>", y_pred[linha])
    print("Resultado real (r)=>", y_test[linha])
print("##################################")

#print(y_pred)
### MÉTRICAS DE DESEMPENHO
erro_m = np.mean((y_test - y_pred) ** 2)
print("Erro quadratico medio:", erro_m)

erro_r = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test - np.mean(y_test)))
print("Erro quadratico relativo:", erro_r)
print("##################################")
