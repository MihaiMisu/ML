from numpy.random import rand
from numpy import dot, array
from scipy.special import expit

class NeuralNetwork:
    
    def __init__(self, 
                 input_nodes=3, 
                 hidden_nodes=3, 
                 output_nodes=3,
                 learning_rate=0.5):
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.learning_rate = learning_rate
        
        self.wih = rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.who = rand(self.output_nodes, self.hidden_nodes) - 0.5
    
    def show_network_features(self):
        print(self.__dict__)
        return self.__dict__
        
    def train(self, inputs, expected_outputs):
        inputs = array(inputs, ndmin=2).T
        expected_outputs = array(expected_outputs, ndmin=2).T
        
        hidden_inputs = dot(self.wih, inputs)
        hidden_inputs = expit(hidden_inputs)
        
        

    def query(self, inputs):
        
        out_hidden = dot(self.wih, inputs)
        out_hidden = expit(out_hidden)
        
        nn_out = dot(self.who, out_hidden)
        nn_out = expit(nn_out)
        
        return nn_out
    
if __name__ == "__main__":
    
    nn = NeuralNetwork(3, 3, 3, 0.3)
    nn.show_network_features()
    print(nn.query([1, 0.5, -1.5]))













