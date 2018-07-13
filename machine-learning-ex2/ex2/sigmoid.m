function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
% this is probably not needed if we implemented our
% own g(x)
% g = zeros(size(z));

%ev = ones(size(z)) .* e;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
% .<op> performs the operation on one-to-one
% Addition and subtraction already supports that.
% Division, multiplication, exponentation needs it 
% explicitly.
g = 1 ./ ( 1 + e.^(-z));


% =============================================================

end
