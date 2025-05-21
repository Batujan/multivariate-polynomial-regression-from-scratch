This is a quadratic multivariate linear regression model made from scratch using only NumPy. The purpose of using a quadratic expansion is to reduce the risk of overfitting, allowing the model to generalize across datasets with varying numbers of features and samples. It is built using a numerical dataset of California housing prices, which lends itself to linear modeling. The program outputs are:

* A 2D plot of predicted vs. actual house prices,

* The R² value, showing how well predictions align with actual values,

* The mean absolute error (MAE), representing the average prediction error in dollars .

The program operates in four main steps: 

1. Loads the dataset from a CSV file, and splits it into a feature matrix X and a target vector y. 

2. Transforms the feature matrix X into a polynomial feature matrix X<sub>poly</sub> by applying the multivariate quadratic expansion: 1+∑<sub>i=1</sub><sup>n</sup>x<sub>i</sub>+∑<sub>i=1</sub><sup>n</sup>x<sup>2</sup><sub>i</sub>+∑<sub>0<i<j=<0</sub><sup>n</sup>x<sub>i</sub>x<sub>j</sub>.

3. Computes the solution vector "x<sup>^</sup>" using the least squares solution formula by taking the left inverse of A. Here's the general equation: x<sup>^</sup>=(A<sup>T</sup>A)<sup>-1</sup>A<sup>T</sup>b. And, here's the version with pseudo-inverses, which is what the program uses (since the feature matrix X has a full rank): x<sup>^</sup>=X<sub>poly</sub><sup>+</sup>y. Finally, the program computes the residuals (errors) in order to evaluate the performance of the model using R squared and mean absolute error values.

4. Plots a 2D graph of actual vs. predicted house prices and prints R squared and MAE values.

**Requirements**

* Python 3
* Matplotlib
* NumPy
* Pandas
