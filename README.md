# CADS: Comparison between Algorithms executed on Datasets and Streams
## Objectives
The aim is to engineer the procedure of Decision Tree evolution based on the
determination of individuals for generations of trees suitable for reproduction
and the selection of stopping criteria.

## Summary
Evolutionary Decision Trees use evolutionary algorithms based on mecha-
nisms inspired by biological evolution to build more robust and efficient de-
cision trees. In order to demonstrate this, a Wine Quality Data Set was
used, from which physico - chemical information relating only to white wine
was considered. Decision Trees refer to a type of classifier that relies on a flow
chart as a tree structure that classifies observations by learning simple deci-
sion rules deduced from the characteristics of the data. Since evolutionary
algorithms represent research heuristics using mechanisms inspired by the
process of natural biological evolution, it was possible to use them to evolve
populations of Decision Trees to achieve the best result. In this context,
the Voting Classifier is a suitable method to simply aggregate the results of
each classifier and thus predict the output class based on the majority of the
highest votes. The evolution of Decision Tree populations has been carried out using a 
selection process whereby only the best individuals (trees) are more likely to reproduce.
Wine Quality Data Set used contains information on two types of wine: white
and red. It was decided to use information on white wine only with the
aim of predicting its quality (with a score from 0 to 10) on the basis of its
physical and chemical properties:
- Input variables (based on physicochemical tests): fixed acidity, volatile
acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total
sulfur dioxide, density, pH, sulphates, alcohol.
- Output variable (based on sensory data): quality.

The use of the Decision Tree on the Wine Quality Dataset is based on a
root node divided into subsets (child nodes) according to specific rules. This
process is repeated on each derived subset until the subset at a node has all
the same values as the target variable, or the subdivision adds no value to
the predictions. The selection process, which ensures that better individuals are more 
likely to reproduce, is represented by the percentage of individuals in a population
that have obtained the highest score (percentage relating to the number of
correct predictions) on the training set. This process is repeated iteratively until the 
stopping criterion represented by the maximum number of generations set in advance is met.
The best represented individual is chosen as the solution.

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

### Authors
*Francesca La Manna, University of Salerno & Gioacchino Caliendo, University of Salerno*

### Professors
*Prof. Fabio Palomba, University of Salerno & Prof. Dario Di Nucci, University of Salerno*

### Tutor
*Ph.D Student Emanuele Iannone*