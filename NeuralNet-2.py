import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_excel(dataFile)

    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        self.processed_data = self.raw_input
        #Convert categorical bean classes to numbers. Bean class names are in Turkish.
        self.processed_data['Class'].replace(['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SIRA'],
                        [0, 1, 2, 3, 4, 5, 6], inplace=True)
        #take a look at the processed data
        print(self.processed_data)

        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)

        #divide attributes (X) from class (Y) which is the target that we want to predict
        self.X = self.processed_data.iloc[:, 0:(ncols - 1)]
        self.y = self.processed_data.iloc[:, (ncols-1)]

        #randomly splits training and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y)

        #Scale data, sklearn documentation says this is important to do for the Multi-Layer Perceptron
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        # Below are the hyperparameters that you need to use for model evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [[6], [2,3]] #array with the ith element representing the # of neurons in the ith hidden layer.

        #total number of networks that will be created
        num_networks = len(activations) * len(learning_rate) * len(max_iterations) * len(num_hidden_layers) #update this if testing with more than one hidden layer configuration
        #keeps track of which network is currently being trained
        network_index = 0

        plot_x_list = [[0] for i in range(num_networks)] #For each network, stores a list of epoch numbers from 1 to the current value of max_iterations
        plot_y_list = [[0] for i in range(num_networks)] #For each network, stores the accuracy for each epoch


        # Create the neural network and be sure to keep track of the performance metrics.
        # This is going to create 3 * 2 * 2 * 2 models, to reduce how long it takes to run, reduce the variety of hyperparameters above.
        for activation in activations:
            for rate in learning_rate:
                for max in max_iterations:
                    for topology in num_hidden_layers:
                        print("Creating neural network with " + str(activation) + " activation function, " + str(rate) + " learning rate, " + str(max) + " max iterations, and hidden layers as follows: " + str(topology))
                        neural_network = MLPClassifier(activation=activations[0], hidden_layer_sizes=topology, alpha=1e-5,
                        learning_rate_init=learning_rate[0], max_iter=max_iterations[1], verbose=False)

                        # Train the model on the training data, running each epoch manually by calling partial_fit each iteration of the for loop, instead of calling fit once
                        for i in range(1, max):
                            #Update the model with a single iteration (epoch)
                            neural_network.partial_fit(self.X_train, self.y_train, np.unique(self.y))
                            #add the current epoch number (x coordinate) to the plot
                            plot_x_list[network_index].append(i)
                            #Get the mean accuracy for this epoch (y coordinate) and add it to the plot
                            accuracy = neural_network.score(self.X_test, self.y_test)
                            plot_y_list[network_index].append(accuracy)

                        # Test the model on the training data
                        y_train_predictions = neural_network.predict(self.X_train)
                        # Test the model on the test data
                        y_test_predictions = neural_network.predict(self.X_test)

                        rmse = (np.sqrt(mean_squared_error(self.y_train, y_train_predictions)))
                        accuracy = neural_network.score(self.X_train, self.y_train)

                        print("Training set performance:")
                        print('RMSE is {}, Accuracy is {}'.format(rmse, accuracy))

                        rmse = (np.sqrt(mean_squared_error(self.y_test, y_test_predictions)))
                        accuracy = neural_network.score(self.X_test, self.y_test)

                        print("Test set performance:")
                        print('RMSE is {}, Accuracy is {}'.format(rmse, accuracy))
                        print("\n")

                        network_index+=1

        # TODO: Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        for i in range(0, num_networks):
            plt.plot(plot_x_list[i], plot_y_list[i], label = "Network " + str(i))
        
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Model history")
        plt.show()

        return 0


if __name__ == "__main__":
    neural_network = NeuralNet("https://haydeni-6375-datasets.s3.us-east-2.amazonaws.com/Dry_Bean_Dataset.xlsx") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
