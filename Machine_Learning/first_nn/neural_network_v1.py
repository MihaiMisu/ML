'''
    Works as expected. Compared to results obtained from GIT source code:
https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork/blob/master/part2_neural_network.ipynb
'''

from numpy.random import normal
from numpy import dot, array, transpose, savetxt, loadtxt
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
        
        self._wih = normal(0.0, pow(self._input_nodes, -0.5), (self._hidden_nodes, self._input_nodes))
        self._who = normal(0.0, pow(self._hidden_nodes, -0.5), (self._output_nodes, self._hidden_nodes))
    
        self.activation_function = lambda x: expit(x)
            
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
        
        hidden_inputs = dot(self._wih, inputs)
        hidden_outputs = expit(hidden_inputs)

        outputs_input = dot(self._who, hidden_outputs)
        outputs_output = expit(outputs_input)
        
        output_errors = expected_outputs - outputs_output
        hidden_errors = dot(self._who.T, output_errors) 
        
        self._who += self._learning_rate * dot((output_errors * outputs_output * (1.0 - outputs_output)), transpose(hidden_outputs))
        
        self._wih += self._learning_rate * dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), transpose(inputs))        

    def query(self, inputs):
        inputs = array(inputs, ndmin=2).T
        
        out_hidden = dot(self._wih, inputs)
        out_hidden = self.activation_function(out_hidden)
        
        nn_out = dot(self._who, out_hidden)
        nn_out = self.activation_function(nn_out)
        
        return nn_out
    
    def query_2(self, inputs_list):
        # convert inputs list to 2d array
        inputs = array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = dot(self._wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = dot(self._who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
    def save_coeffs_to_file(self, file_name,  extension):
        
        savetxt(file_name + "_wih" + extension, self._wih, delimiter=",")
        savetxt(file_name + "_who" + extension, self._who, delimiter=",")
    
    def load_coeffs_from_file(self, wih_file, who_file):
        
        self._wih = loadtxt(wih_file, float, delimiter=",")
        self._who = loadtxt(who_file, float, delimiter=",")
    
if __name__ == "__main__":
    
    nn = NeuralNetwork(3, 3, 3, 0.3)

    nn.show_network_features()
    
    wih = array([
            [-0.3763134 , -0.5486234 ,  0.12163167],
            [ 0.18277464,  0.00545574,  0.40029351],
            [ 0.05494157, -0.30964649,  0.16529363]])
    
    who = array([
            [-0.07645497, -1.54035603,  1.16798749],
            [-1.19954533, -0.1977093 , -0.0460765 ],
            [ 0.21076346,  0.00827052, -0.47213663]])
    
    nn.set_wih(wih)
    nn.set_who(who)
    
    nn.show_network_features()

    git = nn.query([1, 0.5, -1.5])
    print(nn.query([1, 0.5, -1.5]))













