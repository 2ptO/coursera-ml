function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

sumOfSqrdError = 0;

% % given that we have to accumulate the sum only for R(i, j) == 1
% for i = 1 : num_movies
%     for j = 1 : num_users
%         if R(i, j) == 1
%             % In the problem statement, it was given to calculate theta' * X
%             % however, that cannot be used here.
%             % X(i, :) -> 1xN matrix, Theta(j,:) -> 1xN
%             % so X * Theta' will give 1x1 matrix, which is what we want.
%             thisSqrdError = (((X(i, :) * Theta(j,:)') .- Y(i, j)) .^ 2);
%             sumOfSqrdError = sumOfSqrdError + thisSqrdError;
%         end
%     end
% end

% X - movies x features, Theta' - features x users
% Y - movies x users, R - movies x users
J_inclusive_of_all_r = ((X * Theta') .- Y) .^ 2;

J_without_regularization = (1/2) * sum(sum(J_inclusive_of_all_r .* R));

reg_factor_theta = (lambda/2) * sum(sum((Theta .^ 2)));
reg_factor_X = (lambda/2) * sum(sum((X .^ 2)));

J = J_without_regularization + reg_factor_theta + reg_factor_X;

% Compute X_grad using a for loop
% for i = 1 : num_movies
%     idx = find(R(i, :) == 1);
%     Theta_temp = Theta(idx, :); % pick only the users who rated
%     Y_temp = Y(i, idx); % pick only the rates ones
%     X_grad(i, :) = ((X(i, :) * Theta_temp') .- Y_temp) * Theta_temp;
% end

% Compute X_grad and Theta_grad with vectorized method.
% If there was no tutorial for this part, it would have
% been very difficult to complete this. Thanks to the
% course TA.
error_factor = (((X * Theta') .- Y) .* R); % movies x users

grad_reg_factor = (lambda.*X);
theta_grad_reg_factor = (lambda .* Theta);
X_grad = error_factor * Theta + grad_reg_factor;% movies x users * users x features
Theta_grad = error_factor' * X + theta_grad_reg_factor; % movies x users * movies x features

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
