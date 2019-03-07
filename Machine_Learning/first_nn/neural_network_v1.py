'''
    Works as expected. Compared to results obtained from GIT source code:
https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part2_neural_network.ipynb
'''

from numpy.random import normal
from numpy import dot, array, transpose
from scipy.special import expit

class NeuralNetwork:
    
    def __init__(self, 
                 input_nodes=3, 
                 hidden_nodes=3, 
                 output_nodes=3,
                 learning_rate=0.5):
        
        self._input_nodes = input_nodes
        self._hidden_nodes = hidden_nodes
        self._output_nodes = output_nodes
        
        self._learning_rate = learning_rate
        
        self._wih = normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self._who = normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
    
    def set_input_nodes(self, input_nodes):
        self._input_nodes = input_nodes
    
    def set_hidden_nodes(self, hidden_nodes):
        self._hidden_nodes
    
    def set_output_nodes(self, output_nodes):
        self._output_nodes = output_nodes
    
    def set_wih(self, wih):
        self._wih = wih
    
    def set_who(self, who):
        self._who = who
    
    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
    
    def get_input_nodes(self):
        return self._input_nodes
    
    def get_hidden_nodes(self):
        return self._hidden_nodes
    
    def get_output_nodes(self):
        return self._output_nodes
    
    def get_learning_rate(self):
        return self._learning_rate
    
    def get_wih(self):
        return self._wih

    def get_who(self):
        return self._who
    
    def show_network_features(self):
        print(self.__dict__)
        return self.__dict__
        
    def train(self, inputs, expected_outputs):
        inputs = array(inputs, ndmin=2).T
        expected_outputs = array(expected_outputs, ndmin=2).T
        
        hidden_inputs = dot(self.wih, inputs)
        hidden_outputs = expit(hidden_inputs)
        
        outputs_input = dot(self.who, hidden_outputs)
        outputs_output = expit(outputs_input)
        
        output_errors = expected_outputs - outputs_output
        hidden_errors = dot(self.who.T, output_errors) 
        
        self.who += self.lr * dot((output_errors * outputs_output * (1.0 - outputs_output)), transpose(hidden_outputs))
        
        self.wih += self.lr * dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), transpose(inputs))        

    def query(self, inputs):
        inputs = array(inputs, ndmin=2).T
        
        out_hidden = dot(self.wih, inputs)
        out_hidden = expit(out_hidden)
        
        nn_out = dot(self.who, out_hidden)
        nn_out = expit(nn_out)
        
        return nn_out
    
if __name__ == "__main__":
    
    nn = NeuralNetwork(3, 3, 3, 0.3)

    nn.show_network_features()
    
    wih = array([
            [-0.13633719,  0.61624086,  0.54912288],
            [-0.05228535, -0.0593957 ,  0.61747629],
            [-0.33880832,  0.22604528, -0.02312596]
            ])
    
    who = array([
            [-0.26280821, -0.39603203,  0.34228423],
            [-0.58585622,  0.28441407, -0.93537242],
            [-0.57200599,  0.58440066, -0.13354788]
            ])
    
    nn.set_wih(wih)
    nn.set_who(who)
    
    nn.show_network_features()

    git = nn.query([1, 0.5, -1.5])
    print(nn.query([1, 0.5, -1.5]))













