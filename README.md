# CS 6375.502 Assignment 2

## Instr. Anurag Nagar

### Hayden Iske and Leo Bala

### How to run the program
In a terminal, execute:\
python NeuralNet-2.py

Make sure all libraries used, listed below, are installed.\

Results of each model are printed to standard output as well as included in a outputtable.csv file created after the program is run. Each run will update outputtable.csv.\
Before the program ends the plot of model history will appear in a new window. The program ends once you close this plot.

### Results summary
The relu activation function performed the best as we got our highest test accuracies and lowest test errors with this function. The best performing topology of neurons was 1 hidden layer with 6 neurons in it, this topology clearly outperformed the combination of two hidden layers with 2 in the first and 3 in the second layer as evidenced by the model history plot. In the plot the 6 neuron configurations are quicker to converge and converge at higher accuracies than the [2,3] configurations. The best overall combination of hyperparameters was found in #18 in outputtable.csv as it had the highest test accuracy and one of the lowest test errors. #18 used the relu activation function, learning rate of 0.01, max iterations of 200, and hidden layer topology of one hidden layer with 6 neurons. 

### Where is the dataset from?
The dataset was found from the UCI open datasets repository at: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset, credit: KOKLU, M. and OZKAN, I.A., (2020), Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. Computers and Electronics in Agriculture, 174, 105507. Building real estate valuation models with comparative approach through case-based reasoning. Applied Soft Computing, 65, 260-271.

Uploaded to a public s3 bucket, dataset should download to memory automatically when run.

### Libraries and sources used
Provided NeuralNet-2.py file used as starting point.\
sklearn\
numpy\
pandas\
seaborn

### Number of late days used
0
