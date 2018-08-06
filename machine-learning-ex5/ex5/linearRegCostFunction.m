function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% cost function of a linear regression
%
% J(theta) = ((1/(2*m)) * sum(h(x) - y)^2) + (lambda/(2*m)) * sum(theta[2:])^2

% m = length(y)
% h(x) in linear regression
% h(x) = theta*x
% size_of_theta: nx1, size_of_X = mxn
hofX = X * theta;
J_without_theta = (1/(2*m)) * sum((hofX .- y) .^ 2);
reg_parameter = (lambda/(2*m)) * sum(theta([2:end],:) .^ 2);

J = J_without_theta + reg_parameter;

% computing the gradient descent
% grad0 = (1/m) * sum(h(X) - y) * X 
% grad = (1/m) * sum(h(X) - y) + (lambda/m) * theta ; j >= 1 (in Octave terms, it is j >= 2)

grad0 = (1/m) * sum((hofX - y) .* X(:,1));

% theta is of size nx1; grad dimension is 1xn
% Hence we have to take transpose of theta([2:end],:) when calculating
% grad in the below line
grad = ((1/m) * sum((hofX - y) .* X(:,[2:end]))) + ((lambda/m) .* theta([2:end],:)');
grad = [grad0 grad];


% =========================================================================

grad = grad(:);

end
