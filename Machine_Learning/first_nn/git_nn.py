
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
import os

from os.path import dirname, join
from neural_network_v1 import NeuralNetwork as my_net

# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    def get_wih(self):
        return self.wih
    
    def get_who(self):
        return self.who
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

if __name__ == "__main__":

    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    
    # learning rate
    learning_rate = 0.1
    
    # create instance of neural network
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
    n1 = my_net(input_nodes,hidden_nodes,output_nodes, learning_rate)
    #print(n.__dict__)
    #
    #print(n.query([1.0, 0.5, -1.5]))
    
#    n1.set_wih(n.get_wih())
#    n1.set_who(n.get_who())
    
    # load the mnist training data CSV file into a list
    training_file_name = "training_data/mnist_train_100.csv"
    path = dirname(os.getcwd())
    path_to_training_file = join(path, training_file_name)
    training_data_file = open(path_to_training_file.__str__().replace("\\", "/"),
                              'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    
    epochs = 5
    
    for e in range(epochs):
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
            n1.train(inputs, targets)
            
            pass
        pass

    
    # load the mnist test data CSV file into a list
    testing_file_name = "test_data/mnist_test_10.csv"
    path_to_testing_file = join(path, testing_file_name)
    test_data_file = open(path_to_testing_file, 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    
    # test the neural network
    
    # scorecard for how well the network performs, initially empty
    scorecard = []
    
    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        outputs1 = n1.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs1)
        # append correct or incorrect to list
        print("expected: {}; got: {}".format(correct_label, label))
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass
        
    #    break
        pass
    
    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)