function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

yval =  eye(num_labels)(y,:);

% size(X, 1) returns the size of the rows in X
A1 = [ones(size(X, 1),1) X];

% J(theta) = (1/m) * sum(1..m)sum(1..K) [-y * log(h(x)) - (1-y)*log(1-h(x))]
A2 = sigmoid(A1 * Theta1'); % 5000x401 * 401*25 ==> 5000 x 25

A2 = [ones(size(A2, 1), 1) A2];

HofX = sigmoid(A2 * Theta2'); % 5000 x 26 * 26*10 ==> 5000 x 10

% without regularization
J = (1/m) .* sum(sum((-yval .* log(HofX)) - ((1-yval) .* log(1 - HofX))));

size(Theta1);
size(Theta2);

% with regularization
% J(theta) = J_without_regularization + (lambda/(2*m))[sum(theta1) + sum(theta2)]

theta1_sq_sum = sum(sum(Theta1(:,[2:end]) .^ 2));
theta2_sq_sum = sum(sum(Theta2(:,[2:end]) .^ 2));

reg_factor =  (lambda/(2*m)) * (theta1_sq_sum + theta2_sq_sum);

J = J + reg_factor;

% Steps to calculate backpropagation
%
% 1. Forward propagation with Xi
% 2. Small_Delta = (Ak - Yk)

A3 = HofX;

% small delta
% A3 -> mx10, yval->mx10
d3 = A3 - yval;

% theta2 -> 10x26
% 5000x10 * 10x25 = 5000x10
z2 = A1 * Theta1';
d2 = ((d3 * Theta2(:, [2:end]))) .* sigmoidGradient(z2);

delta = 0;
delta1 = (d2' * A1);
delta2 = (d3' * A2);

% delta = delta1 + delta2;

% size(delta1)
% size(delta2)

Theta1_grad = (1/m) * (delta1);
Theta2_grad = (1/m) * (delta2);

% Regularizing theta 
theta1(:,1) = 0;
theta2(:,1) = 0;

theta1_sc  = (lambda/m) * theta1;
theta2_sc = (lambda/m) * theta2;

Theta1_grad = Theta1_grad + theta1_sc;
Theta2_grad = Theta2_grad + theta2_sc;

% size(Theta1_grad)
% size(Theta2_grad)




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
