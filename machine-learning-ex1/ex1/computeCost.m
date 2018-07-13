function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Tried to use theta transpose as taught in the lecture
% learnt that theta' will give 1X2 matrix which cannot
% be used to multiple a 97X2 matrix. Matrix multiplication
% is not commutative.
hx = X * theta;

% . needed for element-wise operation

J = (sum((hx .- y) .^ 2)) / (2 * m)

% =========================================================================

end
