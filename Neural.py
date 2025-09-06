import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 

iris = fetch_ucirepo(id=53) 


class NeuralNet:
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
    
    
    def _randomWeightsHe(self, n_in, n_out):
        return np.random.randn(n_in, n_out) * (np.sqrt(2.0 / n_in) * 0.5)

    def _generateWeight(self):
        self.weights = []
        self.bias = []
        n_in = self.input_features
        for n_out in self.layer_neurons + [self.output_layer]:
            if self.activation_function == 'ReLu':
                W = self._randomWeightsHe(n_in, n_out)
            else:
                W = self._randomWeightsXavier(n_in, n_out)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.bias.append(b)
            n_in = n_out

    #TODO: funcion de regularizcion L1,L2 ,Dropout
    #TODO: more activation functions   
    #TODO: MANEJOR DE activacion por capa

    #derivadas de las funciones de activacion para el gradiente descendente
    def _sigmoid_derivative(self, z):
        sig = self._sigmoid(z)
        return sig * (1 - sig)
    def _ReLu_derivative(self, z):
        return (z > 0).astype(float)
    
    def _sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def _ReLu(self, z):
        return np.maximum(0, z)
    
    def _softmax(self, z):
        # proteger
        z = np.clip(z, -100, 100)
        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
    
    def _forward(self, x):
        """Forward pass que guarda activaciones y valores z"""
        activations = [x]  # Comenzamos con la entrada
        zs = []  # Para almacenar las entradas de cada capa
        
        current_input = x
        
        # Procesar capas ocultas
        for i in range(len(self.layer_neurons)):
            z = current_input @ self.weights[i] + self.bias[i]
            zs.append(z)
            
            # Aplicar funcion de activacin
            if self.activation_function == 'sigmoid':
                current_input = self._sigmoid(z)
            elif self.activation_function == 'ReLu':
                current_input = self._ReLu(z)
                
            activations.append(current_input)
        
        # Capa de salida (sin función de activación, solo softmax al final)
        z_output = current_input @ self.weights[-1] + self.bias[-1]
        zs.append(z_output)
        
        # Aplicar softmax para obtener probabilidades
        y_pred = self._softmax(z_output)
        activations.append(y_pred)
        
        return y_pred, activations, zs

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


    def _backward(self, x, y_true, activations, zs):
        #Viene la funcion de perdida
        m = x.shape[0] # numero de ejemplos
        y_pred = activations[-1]  # salida de la red
        # Convertir y_true a one-hot encoding
        y_true_onehot = np.zeros_like(y_pred)  # matriz de ceros del tamaño de y_pred
        y_true_onehot[np.arange(m), y_true] = 1
        # Calcular delta para la capa de salida
        delta = (y_pred - y_true_onehot)  # diferencia entre prediccion y verdad
        self.gradients_w = []
        self.gradients_b = []

        # Backpropagation a través de las capas de regreso
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]  # activación de la capa anterior

            grad_w = a_prev.T @ delta / m  # gradiente de pesos
            grad_b = np.sum(delta, axis=0, keepdims=True) / m  # gradiente de bias

            self.gradients_w.insert(0, grad_w)
            self.gradients_b.insert(0, grad_b)
            # Calcular delta para la capa anterior
            if i > 0:
                delta = delta @ self.weights[i].T
                # Obtener z de la capa anterior
                z_prev = zs[i-1] # z de la capa anterior
                if self.activation_function == 'sigmoid':
                    # Aplicar la derivada de la función sigmoide 
                    delta = delta * self._sigmoid_derivative(z_prev)
                elif self.activation_function == 'ReLu':
                    delta = delta * self._ReLu_derivative(z_prev)
        # Gradient clipping para evitar explosión de gradientes
        max_norm = 5.0
        for i in range(len(self.gradients_w)):
            n = np.linalg.norm(self.gradients_w[i])
            if n > max_norm:
                self.gradients_w[i] *= max_norm / (n + 1e-8)
        return self.gradients_w, self.gradients_b

                    


    def _updateWeights(self, gradients_w, gradients_b):
        # Actualizar pesos y bias usando gradientes calculados en backward
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.bias[i] -= self.learning_rate * gradients_b[i]



    def fit(self, datos_x: np.ndarray, datos_y):
        # y a índices 
        # aqui estamos asumiendo que y es un vector de etiquetas
        if isinstance(datos_y, pd.Series):
            datos_y = datos_y.to_numpy()
        clases, y_idx = np.unique(datos_y, return_inverse=True) # y_idx son los indices de las clases
        self.classes = clases  # ADDED: guardamos las clases para usar luego en test/predict

        # inicializar pesos
        self.input_features = datos_x.shape[1]
        self.output_layer = len(clases)
        self._generateWeight()

       # Entrenamiento por épocas
        for epoch in range(self.epoch):
            total_loss = 0
            num_batches = 0
            
            # Entrenamiento por mini-batches
            m = datos_x.shape[0]
            indices = np.random.permutation(m)  # Mezclar datos
            
            for i in range(0, m, self.batch_size):
                # Obtener batch
                batch_indices = indices[i:i + self.batch_size]
                x_batch = datos_x[batch_indices]
                y_batch = y_idx[batch_indices]
                
                # Forward pass
                y_pred, activations, zs = self._forward(x_batch)
                
                # Calcular pérdida
                batch_loss = self._CategoricalCrossEntropy(y_batch, y_pred)
                total_loss += batch_loss
                num_batches += 1
                
                # Backward pass
                gradients_w, gradients_b = self._backward(x_batch, y_batch, activations, zs)
                
                # Actualizar pesos
                self._updateWeights(gradients_w, gradients_b)
            
            # Calcular métricas promedio de la época
            avg_loss = total_loss / num_batches
            
            # Evaluar en todo el dataset cada 10 épocas
            if (epoch + 1) % 10 == 0 or epoch == 0:
                y_pred_full, _, _ = self._forward(datos_x)
                pred_labels = np.argmax(y_pred_full, axis=1)
                acc = np.mean(pred_labels == y_idx)
                print(f"Epoch {epoch+1:3d} | loss={avg_loss:.4f} | acc={acc:.3f}")




    def predict(self, datos_x :np.ndarray):
        y_pred, _, _ = self._forward(datos_x)
        return np.argmax(y_pred, axis=1) # de todas las predicciones escoge la de mayor probabilidad



def stratified_split(X, y, test_size=0.3, seed=42):
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y)
    if y.ndim > 1:  # aplastar etiquetas para que sea 1D
        y = y.ravel()
    classes, y_idx = np.unique(y, return_inverse=True)
    train_idx = []
    test_idx = []
    for c in range(len(classes)):
        idx_c = np.where(y_idx == c)[0]
        rng.shuffle(idx_c)
        n_test = int(len(idx_c) * test_size)
        test_idx.extend(idx_c[:n_test])
        train_idx.extend(idx_c[n_test:])
    train_idx = np.array(train_idx); test_idx = np.array(test_idx)
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]



#para probar nomas
modelo1 = NeuralNet(learning_rate=0.01, 
                  activation_function='ReLu', 
                  layer_neurons=[8,4], 
                  epoch=200, 
                  batch_size=16)

X = iris.data.features.to_numpy()
y = iris.data.targets.to_numpy()
if y.ndim > 1:   # aplastar etiquetas
    y = y.ravel()


# escalar para que tenga media 0 y desviacion estandar 1
X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.3, seed=7)

modelo1.fit(X_train, y_train)

y_pred = modelo1.predict(X_test)
# mapear y_test a índices de train
class_to_idx = {c:i for i,c in enumerate(modelo1.classes)}
y_test_idx = np.array([class_to_idx[label] for label in y_test])

test_acc = np.mean(y_pred == y_test_idx)
print(f"Accuracy test: {test_acc:.3f}")
