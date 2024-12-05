#Displaying Lmplots
#lmplot is scatter plot

import seaborn as sns # biblioteca para visualização de dados baseada em Matplotlib
import matplotlib.pyplot as plt # a biblioteca Matplotlib fornece funções para criar gráficos em Python.

#---load the iris dataset---
iris = sns.load_dataset("iris") # Esse dataset contém dados sobre o comprimento e largura de pétalas e sépalas de três espécies de flores Iris.

#---plot the lmplot---
# sns.lmplot: cria um gráfico de dispersão (scatter plot) com base em duas variáveis contínuas
sns.lmplot('petal_width', 'petal_length', data=iris,
           hue='species', palette='Set1',
           fit_reg=False, scatter_kws={"s":70})
# 'petal_width': especifica o eixo x do gráfico (largura da pétala).
# 'petal_length': especifica o eixo y do gráfico (comprimento da pétala).
# data=iris: indica o dataset a ser usado (neste caso, o dataset Iris).
# hue='species': a cor dos pontos no gráfico será diferenciada pela espécie da flor (Iris Setosa, Versicolor ou Virginica).
# palette='Set1': define a paleta de cores para diferenciar as espécies no gráfico.
# fit_reg=False: desativa a linha de regressão (por padrão, lmplot inclui uma linha de regressão linear).
# scatter_kws={"s":70}: especifica o tamanho dos pontos no gráfico (aqui os pontos terão um tamanho de 70).

#---get the current polar axes on the current figure---
ax = plt.gca() # "Get Current Axes" — Obtém o eixo atual do gráfico (ou seja, o sistema de coordenadas onde o gráfico está sendo plotado).
ax.set_title("Ploting using the Iris dataset") # adiciona um título ao gráfico.

#---show the plot---
plt.show() # exibe o gráfico gerado.