function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X1 = [ones(size(X, 1), 1) X];

A = sigmoid(X1 * Theta1'); % 5000x401 * 401*25 ==> 5000 x 25

A1 = [ones(size(A, 1), 1) A];

HofX = sigmoid(A1 * Theta2'); % 5000 x 26 * 26*10 ==> 5000 x 10

[val, p] = max(HofX, [], 2); %max(X, [], DIM), DIM = 1 if column, 2 if row












% =========================================================================


end
