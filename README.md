<div id="top"></div>

<!-- PROJECT -->
<br />
<div align="center">
    <img src="img/methodology.PNG" alt="Methodology" width="80" height="80">
 
<h3 align="center">AEGADT: Application and Evaluation of Genetic Algorithms on Decision Trees</h3>

  <p align="center">
    The project at the centre of this project aims to apply and evaluate the evolution of Decision Trees by means of a Genetic Algorithm. The objectives of the project emphasise the desire to compare the results of different runs of equivalent models, obtained through a batch methodology, to the chosen Genetic Algorithm. Identifying the rate of improvement of decision trees and the number of generations required to obtain the maximum number of correct predictions is an interesting step that this project aims to achieve. It is also important to make observations on the application of the two techniques, considering various factors on the results obtained in the various runs.
    <br />
    <a href="https://github.com/LamFra/doc"><strong>Explore the documentation»</strong></a>
    <br />
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#requirements">Requirements</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contacts">Contacts</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <ul>
        <li><a href="#datasets">Datasets</a></li>
      </ul>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
The focus of this project is on the application and evaluation of Genetic Algorithms on Decision Trees. 

In order to achieve this goal, we propose:
1. The implementation of a Genetic Algorithm capable of operating on decision trees;
2. The application of the Genetic Algorithm using different configurations;
3. The construction of Decision Trees by means of batch learning techniques based on the same configurations;
4. The comparison of the results obtained between equivalent models.

In this project we propose to evaluate the functioning of a Genetic Algorithm in training populations of Decision Trees in order to analyse the
results obtained by having one Decision Tree constructed using a classical training technique called CART (Classification And Regression Trees), and
another constructed on the basis of the reproduction of the best individual in each generation. In this regard, it becomes interesting to empirically identify
the rate of improvement of Decision Trees and the number of generations required to reach the maximum number of correct predictions made by a Decision Tree. 

All the details regarding the project are available [here](doc/AEGADT.pdf). 

<p align="right">(<a href="#top">back to top</a>)</p>

### Requirements

* Numpy  (recommended version 1.21.5 )
* Matplotlib (recommended version 3.5.1)
* Scikit-learn (recommended version 1.0.2)

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With
* Windows 11
* Windows 10

<!-- GETTING STARTED -->
## Getting Started

### Installation
Install the module

Recommended way:

1. Clone the repository
   ```sh
   git clone https://github.com/LamFra/AEGADT.git
   ```
2. Install the requirements
   ```sh
   pip install numpy==1.21.5
   ```
   
   ```sh
   pip install matplotlib==3.5.1
   ```
   
    ```sh
   pip install scikit-learn==1.0.2
   ```
3. Run GA.py
    ```sh
    python GA.py
   ```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contacts

Francesca La Manna - f.lamanna5@studenti.unisa.it <br>
Gioacchino Caliendo - g.caliendo16@studenti.unisa.it

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

###Datasets

* [Wine Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)

<p align="right">(<a href="#top">back to top</a>)</p>

