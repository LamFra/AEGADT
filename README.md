# CTGA: Comparison between Traditional and Genetic training Algorithms
## Objectives
The aim of this paper is to use evolutionary algorithms, such as genetic algorithms, to grow decision trees without the 
use of classical learning algorithms,
e.g., CART or ID3.
## Proposed Methodology
The use of the Decision Tree on a dataset is based on a root node divided into
subsets (child nodes) according to specific rules. This process is repeated on
each derived subset until the subset at a node has all the same values as the
target variable, or the subdivision adds no value to the predictions.
* **Structure of Chromosomes / Individuals**: The structure of Chromosomes/ Individuals is represented by a *Complete 
  Binary Tree* with a maximum depth of 4 was used.
* **Fitness function**: The fitness function was achieved with a close correlation of these metrics: *f-measure, 
  precision and recall*.
* **Selection rule**: The *Roulette wheel selection* method was chosen. 
  This method is a stochastic selection method, where the probability for selection of an individual is proportional to 
  its fitness.
* **Crossover rule**: The *One Point Crossover* was chosen. In this method one arbitrary combination point is selected 
  for both parents’ chromosomes. The chromosomic section after these combination points are swapped with each other, 
  giving birth to two new offspring.
* **Mutation rule**:  The *Flip Bit Mutation* was chosen. This method represents a random change in a chromosome 
  to introduce new patterns to a chromosome.
* **Objective function**: The metric used to determine the best way to generate a Decision Tree is *Information Gain.*

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

### Authors
*Francesca La Manna, University of Salerno & Gioacchino Caliendo, University of Salerno*

### Professors
*Prof. Fabio Palomba, University of Salerno & Prof. Dario Di Nucci, University of Salerno*

### Tutor
*Ph.D Student Emanuele Iannone*