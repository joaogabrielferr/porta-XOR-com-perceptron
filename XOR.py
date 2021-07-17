import numpy as np

class Perceptron:
    
    def __init__(self,treino,target):
        self.entrada = treino
        self.saida = target
        self.no_entradas = len(entrada[0])
        self.pesos = np.zeros(self.no_entradas + 1)
        self.epocas=100
        self.learning_rate=0.01

    def predict(self,e,p):
        #np.dot produto dos vetores a x b
        soma=0
        #print(p[:no_entradas])
        soma = np.dot(e,p[:self.no_entradas])
        #ao final soma o bias    
        soma = soma + self.pesos[self.no_entradas]
        if soma > 0:
            activacao = 1
        else:
            activacao = 0            
        return activacao

    def train(self):
        for _ in range(self.epocas):
            xy =zip(entrada, self.saida) 
            for inputs, label in xy:
                prediction = self.predict(inputs,self.pesos)
                self.pesos[0] += self.learning_rate * (label - prediction) * inputs[0]
                self.pesos[1] += self.learning_rate * (label - prediction) * inputs[1]
                self.pesos[2] += self.learning_rate * (label - prediction)
    

entrada = []
entrada.append(np.array([1, 1]))
entrada.append(np.array([1, 0]))
entrada.append(np.array([0, 1]))
entrada.append(np.array([0, 0]))

saidaand = np.array([1, 0, 0, 0])
saidaxor = np.array([0,1,1,0])
saidaor = np.array([1,1,1,0])
saidanand = np.array([0,1,1,1])

_or = Perceptron(entrada,saidaor)
_nand = Perceptron(entrada,saidanand)

resultadoNAND = []
resultadoOR = []
entradafinal = []

_nand.train()
for i in range(4):
    resultadoNAND.append(_nand.predict(_nand.entrada[i], _nand.pesos))

_or.train()
for i in range(4):
    resultadoOR.append(_or.predict(_or.entrada[i],_or.pesos))

for i in range(4):
    entradafinal.append(np.array([resultadoNAND[i],resultadoOR[i]]))

_xor = Perceptron(entradafinal,saidaand)
_xor.train()

print("final:")
for i in range(4):
    print(_xor.predict(_xor.entrada[i],_xor.pesos))
