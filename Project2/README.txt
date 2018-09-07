
The Code
The code was written in Java and used the ABAGAIL library which has
support for Randomized Optimization Algorithms. 

The code can be built as an Intelli-J project. You can open the project in
Intell-J and build it and run it in Intelli-J. When opening the code in Intelli-J, click on “Open Existing Project” then click on the folder “RandomizedOptimization”. Make sure that you have the ABAGAIL.jar file as a dependency of the project. The ABAGAIL.jar file should be submitted with the code


CSV files and Graphs of the results are under RandomizedOptimization/src/data/results

Dataset
White Wine Quality Dataset
https://archive.ics.uci.edu/ml/datasets/wine+quality
The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are munch more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.


Attribute Information:

Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
Output variable (based on sensory data):
12 - quality (score between 0 and 10)
