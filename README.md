# InLineCoordinatesCoefficientSolver

In-Line Coordinates Coefficient Solver workspace while implementing novel coefficient analysis technique.

## Demonstrative Screenshot

First plotting version of the In-Line Coordinates Coefficient Solver.
![Demo 1](screenshots/demo1.png)

Demo 1 Data set:  

Class 1:  
    - Sample 1: [1.76405235 0.40015721 0.97873798 2.2408932 ]  
    - Sample 2: [ 1.86755799 -0.97727788  0.95008842 -0.15135721]  
    - Sample 3: [-0.10321885  0.4105985   0.14404357  1.45427351]  

Class 2:  
    - Sample 1: [0.76103773 0.12167502 0.44386323 0.33367433]  
    - Sample 2: [ 1.49407907 -0.20515826  0.3130677  -0.85409574]  
    - Sample 3: [-2.55298982  0.6536186   0.8644362  -0.74216502]

## TASK for the CoefSwap "Coefficient Swapping" project

We are interested in the following tasks:

0. Plot the data, find issue where need to fix final swap order of two cases.
1. Fix one case coefficient.
2. Solve for the next case coefficient.
3. Replot the data, check if cases endpoint is swapped correctly.

We have the scenario where we have a set of data points, and we want to find the coefficients for a linear combination that swaps two conflicted cases of either class in the data.

## How to run the code

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Run the code:

```bash
python test.py
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
