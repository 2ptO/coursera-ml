function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

fprintf ('Size of Centroids: %d\n', size(centroids))
fprintf ('Size of X(1) %d\n', size(X))
fprintf ('Size of idx %d\n', size(idx))

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for i = 1:size(X, 1)
    % idx(i) = j that minimizes ||Xi - Muj||^2
    min_C = inf;
    min_j = size(centroids, 1);
    % Going with a iterative solution for now.
    % I'm sure there is a vectorized solution
    for j = 1:size(centroids, 1)
        Cj = sum((X(i, :) .- centroids(j, :)) .^ 2);
        if Cj < min_C
            min_C = Cj;
            idx(i) = j;
        endif
    end
    
end





% =============================================================

end

