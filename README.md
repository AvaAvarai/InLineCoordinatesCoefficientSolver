# InLineCoordinatesCoefficientSolver

Dynamic In-Line Coordinates Coefficient Solver is a workspace for implementing new coefficient swapping technique of linear combinations of data points which are in conflict in their class label as plotted in Dynamic In-Line Coordinates [Kovalerchuk et al. 2018] [Williams and Kovalerchuk manuscript 2024].

## Demonstrative Screenshot

First plotting version of the In-Line Coordinates Coefficient Solver.
![Demo 1](screenshots/demo1.png)

## Data Set Used

We are using artificial data set for exploratory analysis of the new coefficient analysis technique and to validate the technique in a simple scenario.

Data is loaded from `data.csv` and plotted on a 1D subspace (x-axis or y=0). Found and directly editted in `data.csv` to be used for testing.

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
python main.py
```

## License

This project is free to use for both personal and commercial uses as licensed under the MIT License. See the `LICENSE` file for full details.
