# Bayesian Machine Learning Algorithms

![Language](https://img.shields.io/badge/language-Java-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Project Description

This repository contains **Java implementations** of fundamental Bayesian Machine Learning algorithms. These programs demonstrate how probabilistic models work from scratch, without relying on high-level ML libraries.

The project covers key concepts such as belief networks, regression, expectation-maximization, and classification, providing a low-level look at the mathematics behind the models.

### Key Algorithms Implemented
* **Bayesian Belief Networks (BBN):** Models conditional dependencies between variables using a directed acyclic graph (DAG).
* **Bayesian Linear Regression:** Performs regression analysis using Bayesian inference to estimate probability distributions for model parameters.
* **Expectation-Maximization (EM) Algorithm:** Iteratively finds maximum likelihood estimates for parameters in statistical models (often used for clustering or missing data).
* **Naive Bayes Classifier:** A probabilistic classifier based on Bayes' theorem, tested here using the `spambase.csv` dataset.

---

## Project Structure

* **`BayesianBeliefNetworks.java`**: Implementation of BBN logic.
* **`BayesianLinearRegression.java`**: Logic for Bayesian regression.
* **`EMAlgorithm.java`**: Implementation of the Expectation-Maximization algorithm.
* **`NaiveBayesDriver.java`** & **`NaiveBayesClassifier.java`**: The main driver and logic for the Naive Bayes classifier.
* **`MultivariateGaussianPDF.java`**: Helper class for handling Gaussian Probability Density Functions.
* **`spambase.csv`**: Dataset used for testing the Naive Bayes classifier.

---

## 🛠️ Technologies Used

* **Language:** Java (JDK 8 or higher recommended)
* **Standard Library:** Uses standard Java I/O and Math libraries.

---

## Getting Started

### Prerequisites

Ensure you have the Java Development Kit (JDK) installed:
* **javac** (Compiler)
* **java** (Runtime)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YuvrajDoshi01/Bayesian-ML.git
    cd Bayesian-ML
    ```

---

## Compilation & Build Instructions

Since these are standard Java files, you can compile all of them at once using the Java compiler.

**Compile all source files:**
```bash
javac *.java
```

*This will generate the corresponding `.class` bytecode files in the same directory.*

---

## Usage

There are four main executable classes in this project. You can run them individually after compilation.

### 1. Bayesian Belief Networks
Examines conditional dependencies.
```bash
java BayesianBeliefNetworks
```

### 2. Bayesian Linear Regression
Runs the regression model.
```bash
java BayesianLinearRegression
```

### 3. EM Algorithm
Executes the Expectation-Maximization logic.
```bash
java EMAlgorithm
```

### 4. Naive Bayes Classifier
Runs the classifier on the provided spam dataset.
```bash
java NaiveBayesDriver
```
*Note: Ensure `spambase.csv` is in the same directory so the driver can find it.*

