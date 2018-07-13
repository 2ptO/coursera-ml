function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

% we now use the optimized theta values
% for a given x, find h(x)
% if h(x) >= 0.5, prediction = 1 else 0
% I'm confused about one thing..
% theta dimension is nx1
% X dimension is mxn
% Transpose(theta) dimension = 1xn
% how can I multiply 1xn * m*n

hofx = sigmoid(X * theta)
p = hofx >= 0.5






% =========================================================================


end
