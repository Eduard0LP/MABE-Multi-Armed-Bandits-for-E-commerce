# Multi-Armed Bandits for E-commerce (MABE)

The Python code in this repository was developed as part of my master's thesis. The thesis consisted on an introduction to the Multi-Armed Bandit problem and how the solutions to this problem can be applied in real world scenarios, such as the optimization of the recommendations of an e-commerce platform.  

The repository is structured in the following way:  

- mabe: This folder contains the main code of the repository and it is structured as a Python package where all the Python functions and classes in the child folders can be imported. This folder have 2 child folders:

    - models: This folder contains the implementation of the Epsilon-Greedy, Upper Confidence Bound, Thompson Sampling, Linear Upper Confidence Bound and Linear Thompson Sampling algorithms.
    - simulator: This folder contains a simulator/bot that generates data that could be used to train the models of the models folder. It contains 3 main functions: generate_user, that can be used to generate user of an e-commerce platform; generate_products to generate the products; and simulate_interactions, which takes the user and products generated and simulates if a user would click on a product if it was suggested to him by the platform. For more information about the inner workings of the simulator, please take a look at the documentation.

- examples: This folder contains Jupyter notebooks of how the models can be used in combination with the simulator. There are 5 Jupyter notebooks in this folder.  
epsilon_greedy_hyperparameter_tuning, ucb_hyperparameter_tuning, linucb_hyperparameter_tuning and lints_hyperparameter_tuning are examples of how the hyperparameters of the models developed could be optimized for the case of tthe simulator developed.  
training_example contains an example of how the different models colud be trained and compared all together.

## Dependencies
The core of the package only requires numpy and pandas.  

For the execution of the example Jupyter notebooks, matplotlib, seaborn and scikit-learn are also required.

## Installation
The package has not been uploaded to PyPI, as I don't believe it's content is relevant enough to have a broad use from the community.  

In order to proceed with the installation, a clone of the repository has to be done. With the repository clone execute the following command in the root folder of the cloned repository in a terminal with pip available.

```bash
pip install .
```