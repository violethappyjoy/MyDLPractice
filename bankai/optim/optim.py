import numpy as np
import math
'''
Adam Optimizer:
'''
class Adam():
    def __init__(self, parameters, eta=0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) -> None:
        '''
        Initializes v and m as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
         Arguments:
         parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
         eta -- value for eta(default 0.01)
         beta1 -- value for beta1(default 0.9)
         beta2 -- value for beta2(default 0.999)
         epsilon -- value for learning rate (default 1e-8)
        '''
        self.L = len(parameters)//2
        self.m={}
        self.v={}
        
        for i in range(self.L):
            self.m["dw"+str(i+1)] = np.zeros(parameters["W" + str(i+1)].shape)
            self.m["db" + str(i+1)] = np.zeros(parameters["b" + str(i+1)].shape)
            self.v["dW" + str(i+1)] = np.zeros(parameters["W" + str(i+1)].shape)
            self.v["db" + str(i+1)] = np.zeros(parameters["b" + str(i+1)].shape)
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 0
        
    def update(self, parameters, grads):
        '''
        Update parameters using Adam
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
        '''
        self.t += 1
        
        for l in range(self.L):
            # Update moving averages for gradients
            self.m["dw" + str(l+1)] = self.beta1 * self.m["dw" + str(l+1)] + (1 - self.beta1) * grads["dW" + str(l+1)]
            self.m["db" + str(l+1)] = self.beta1 * self.m["db" + str(l+1)] + (1 - self.beta1) * grads["db" + str(l+1)]

            # Update moving averages for squared gradients
            self.v["dW" + str(l+1)] = self.beta2 * self.v["dW" + str(l+1)] + (1 - self.beta2) * (grads["dW" + str(l+1)] ** 2)
            self.v["db" + str(l+1)] = self.beta2 * self.v["db" + str(l+1)] + (1 - self.beta2) * (grads["db" + str(l+1)] ** 2)

            # Bias correction for moving averages
            self.m["dw" + str(l+1)] /= (1 - self.beta1 ** self.t)
            self.m["db" + str(l+1)] /= (1 - self.beta1 ** self.t)
            self.v["dW" + str(l+1)] /= (1 - self.beta2 ** self.t)
            self.v["db" + str(l+1)] /= (1 - self.beta2 ** self.t)

            # Update parameters
            parameters["W" + str(l+1)] -= self.eta * self.m["dw" + str(l+1)] / (np.sqrt(self.v["dW" + str(l+1)]) + self.epsilon)
            parameters["b" + str(l+1)] -= self.eta * self.m["db" + str(l+1)] / (np.sqrt(self.v["db" + str(l+1)]) + self.epsilon)

        return parameters, self.v, self.m
        