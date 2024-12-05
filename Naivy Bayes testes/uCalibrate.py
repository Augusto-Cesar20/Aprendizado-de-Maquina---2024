from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV # usada para ajustar a probabilidade prevista por um classificador. Isso permite calibrar os scores de probabilidade

iris = datasets.load_iris()
features = iris.data 
target = iris.target

classifer = GaussianNB() 

# Create calibrated cross-validation with sigmoid calibration
#irá calibrar as probabilidades previstas pelo classificador Naive Bayes utilizando uma técnica de validação cruzada com 2 divisões (cv=2).
classifer_sigmoid = CalibratedClassifierCV(classifer, cv=2, method='sigmoid')
# sigmoid, que é baseado na regressão logística e é útil para ajustar a confiança das previsões em classificadores que podem ser "overconfident". 
# A validação cruzada divide os dados em 2 subconjuntos, treina o classificador em um subconjunto e testa no outro, garantindo que a calibração seja robusta.

# Calibrate probabilites
classifer_sigmoid.fit(features, target)

# Create new_observation
new_observation = [[2.6, 2.6, 2.6, 0.4]]

# View calibrated probabilities
print(classifer_sigmoid.predict_proba(new_observation)) # exibirá um array onde cada valor representa a probabilidade ajustada de que a nova observação pertença a cada uma das classes (espécies de flores).

