function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% fprintf('size of theta is %f\n', size(theta));
% fprintf('size of x is %f\n', size(X));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% cost function of logistic regression without regularization
% J(theta) = 1/m * sum ( -y*log(h(x)) - (1-y)log(1-h(x)))

% cost function of logistic regression with regularization
% adds the regularization parameter lambda as well
% J(theta) = 1/m * sum ( -y*log(h(x)) - (1-y)log(1-h(x)))

% hofx = (1 ./ (1 + e .^ (-(X * theta))));
hofx = sigmoid(X*theta);

% I first added theta(0) as well in cost calculation
% submit was failed and then realized the mistake.
J = ((1/m) * sum((-y .* log(hofx)) - ((1 - y) .* log(1 - hofx)))) + ((lambda/(2*m)) * sum(theta([2:end],:) .^ 2));

% while computing the gradient of the cost function, we should
% not regularize theta-0 (which is theta(1) in octave)
% so compute J0 separately and then compute the remaining
% values. It took me a long time to figure out the proper
% sequence of operations. Attention to formula indices
% is very important.

j0 = ((1/m) * sum((hofx - y) .* X(:,1)));
% have to take transpose of the regularization parameter (lambda/m)*theta
% Otherwise dimensions won't match for addition.
grad = ((1/m) * sum((hofx - y) .* X(:,[2:end]))) + ((lambda/m) .* theta([2:end],:))';

grad = [j0 grad];









% =============================================================

end
