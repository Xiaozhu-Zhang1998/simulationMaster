
# Simulation Master
  
This package builds a powerful Shiny App for people interested in statistical learning simulation, specifically for the regression task for sparse data and the classification task for data without linear decision boundary. We allow users to generate all different types of data and adjust model parameters, so they will have a direct understanding about their actions' influences through the corresponding changes in resulting visualization graphs.

***Please read the vignette document for any details or demo of this Package/ Shiny App.***


## Package Dependency and Installation

This package depends on a package `bestsubset` released on GitHub, so please install it by running the following code:
```
library(devtools)
install_github(repo = "ryantibs/best-subset", subdir = "bestsubset")
```

Then you can install and load our `simlationMaster` package by running the following code:
```
install_github(repo = "Sta523-Fa21/final_proj_simulation_master", 
               subdir = "simulationMaster")
library(simulationMaster)
```

## Function(s)
This package only has one function `run_simulation_master()`, which is a function to trigger the Shiny App, so it bears no parameters or values. In order to run this Shiny App, please run the following code:
```
run_simulation_master()
```

## Tasks

### Regression

For regression with `cv.glmnet`, **data generation parameters** and **model specification parameters** can be tuned and adjusted freely. We allow the user to select a large bunch of parameters at one time. Then, our function will generate desired data and fit the model. The corresponding results will be shown accordingly.

The user is able to see the **training MSE** and **cross-validation MSE**, and check the **feature selection** results. We also provide a summary of the regression. With the interactive and straightforward results shown in these panels, users will be able to quickly grasp the information they need and further tune and compare the regression models.


### Classification

For classification with kernel SVM, similar to the regression playground, the tuning panels include two parts, **data generation** and **model specification**. Users can choose different data types and other data generation parameters on the left panel and choose a specific kernel and its corresponding parameters for kernel SVM on the right panel to check the results under the desired conditions.

## R CMD check
1. Since this is a package for only 1 function to trigger the Shiny App, we do not include any tests.

2. This package passes the `R CMD check`, i.e. there are no Errors, Warnings, or Notes. It is notable that we use the standard evaluation version of functions in the package `dplyr` to avoid any notes.

## Authors

- Xiaozhu Zhang
- Xinran Song
- Tong Lin
- Xuyang Tian

