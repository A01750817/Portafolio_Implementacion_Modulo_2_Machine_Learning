import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 

iris = fetch_ucirepo(id=53) 


class Network:
    def __init__(self, learning_rate, activation_function :str, layer_neurons: list, epoch: int, batch_size: int):
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.layer_neurons = layer_neurons
        self.input_features = None
        self.output_layer = None
        self.epoch = epoch
        self.batch_size = batch_size
    
    def _randomWeightsXavier(self,  n_in, n_out):
        limit = np.sqrt( 6 / (n_in + n_out))
        weights = np.random.uniform(-limit, limit, (n_in, n_out))
        return weights

    def _generateWeight(self):
        # entre el layer de entrada y el primer oculto es in 4 out 8
        self.weights = []
        self.bias = []
        n_in = self.input_features
        for n_out in self.layer_neurons + [self.output_layer]: #sumamos la capa de salida para ligar la ultima capa oculta con la de salida
            W =self._randomWeightsXavier(n_in, n_out)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.bias.append(b)
            n_in = n_out

    
    #TODO: funcion de regularizcion L1,L2 ,Dropout
    #TODO: more activation functions   
    #TODO: MANEJOR DE activacion por capa
    
    def _sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def _ReLu(self, z):
        return np.maximum(0, z)
    
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # estabilidad numérica
        return exp_z / np.sum(exp_z, axis=1, keepdims=True) # normalización
        
    
    def _forward(self, x):
        #Los datos pasan por la red, capa por capa:
        current_input_layer = x # el primer input layer es el dataset
        for i in range(len(self.layer_neurons) + 1):
            w = self.weights[i]
            
            z = current_input_layer @ w + self.bias[i]
            # aplicar función de activación aquí
            if self.activation_function == 'sigmoid':
                current_input_layer = self._sigmoid(z)
            elif self.activation_function == 'ReLu':
                current_input_layer = self._ReLu(z)
        #z_out corresponde a la ultima capa, osea los resultados
        Z_out =  current_input_layer @ self.weights[-1] + self.bias[-1]

        y_pred = self._softmax(Z_out)
        #ultimo output es la predicción
        return y_pred

    def _MiniBatchGradientDescent(self, x, y):
        batch_size = self.batch_size
        m = x.shape[0]
        for i in range(0, m, batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            self._forward(x_batch)

    def _CategoricalCrossEntropy(self, y_true_idx, y_pred):
        m = y_true_idx.shape[0]
        eps = 1e-12
        probs = np.clip(y_pred[np.arange(m), y_true_idx], eps, 1.0)
        return -np.mean(np.log(probs))


    def _backward(self, x, y_true, y_pred):
        #Viene la funcion de perdida
        pass

    def _updateWeights(self):
        pass



    def fit(self, datos_x: np.ndarray, datos_y):
        # y a índices (0..C-1)
        # aqui estamos asumiendo que y es un vector de etiquetas
        if isinstance(datos_y, pd.Series):
            datos_y = datos_y.to_numpy()
        clases, y_idx = np.unique(datos_y, return_inverse=True)

        self.input_features = datos_x.shape[1]
        self.output_layer = len(clases)
        self._generateWeight()

        #foward + backward + update
        for epoch in range(self.epoch):
            y_pred = self._forward(datos_x)
            loss = self._CategoricalCrossEntropy(y_idx, y_pred)
            if (epoch+1) % 10 == 0 or epoch == 0:
                pred_labels = np.argmax(y_pred, axis=1)
                acc = np.mean(pred_labels == y_idx)
                print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | acc={acc:.3f}")
            # TODO: backward + update aquí


    def predict(self, datos_x :np.ndarray):
        y_pred = self._forward(datos_x)
        return np.argmax(y_pred, axis=1)



modelo1 = Network(0.1, 'sigmoid', [8,4], 50, 32)
#preparar datos
x = iris.data.features.to_numpy()
y = iris.data.targets
#dividir en train y test
x_train = x[100:]
y_train = y[100:]
modelo1.fit(x_train,y_train)