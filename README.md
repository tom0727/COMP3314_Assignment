# COMP3314_Assignment2
***
Group members: Hu Zhenwei(3035533719)

In this assignment, I implement a Convolutional neural network with 7 layers, the model is tested on the dataset MNIST.
***

#### How to run code <br>

Method1: Through Terminal <br>
Under the directory ```assignment2/``` , run the following command in the terminal:
``` python3 A2.py ```

Method2: Through IDE <br>
Open an IDE and import project if necessary, if you are using Pycharm, open A2.py and just run the code is sufficient.

It will first train the model, if during the first epoch the model has no sign of convergence (that is, the loss is around 580-600 all the time), then consider running the program again to try a different weight initialization. The possibility of convergence is low and it might take several tries to have it successfully converges.

The results and analysis are written in the report.

<br><br><br><br>

# COMP3314_Assignment1
***
Group members: Hu Zhenwei(3035533719) & Chang Liyan (3035534880)

In this assignment, we implement SVM algorithm to classify the given datasets, i.e., breast cancer dataset and iris dataset.
***
#### How to run code <br>

Method1: Through Terminal <br>
Under the directory ```assignment1/``` , run the following command in the terminal:
``` python3 SVM.py ```

Method2: Through IDE <br>
Open an IDE and import project if necessary, if you are using Pycharm, open SVM.py and just run the code is sufficient.

It will first train the model for breast cancer dataset and test it with different values of C afterwards. Then, it repeats again for iris data set.  
We adjust the value of slack variable to see how the accuracy will change accordingly. A visual of accuracy of two datasets agianst values of C is plotted.  
The results and analysis are written in the report.
