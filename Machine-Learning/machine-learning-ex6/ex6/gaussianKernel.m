function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

% Guassian kernel = e ^ (- ||Xi - Xj||^2 / 2*sigma^2)
% || Xi - Xj ||^2 = sum((xi - xj))^2

norm = sum((x1 .- x2) .^ 2);
sim = e ^ ( -norm / (2 * sigma^2));

% Some details on the similarity function and the paramter sigma
% think of the Gaussian kernel as a sim- ilarity function that
% measures the “distance” between a pair of examples, (x(i),x(j)).
% The Gaussian kernel is also parameterized by a bandwidth parameter, σ, 
% which determines how fast the similarity metric decreases (to 0) as
% the examples are further apart


% =============================================================
    
end
