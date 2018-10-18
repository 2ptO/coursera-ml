function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% find returns a vector of nonzero elements of a matrix
% here, y is result loaded from the data set.
% we find the marks with y==1 and y==0 and load them
% into separate vectors
pos = find(y==1);
neg = find(y==0);

% Plotting the example data set
% k+ -> use + symbols to plot the data
% k0 -> use circles to plot the data
% MarkerFaceColor 'y' stands for yellow
plot(X(pos, 1), X(pos, 2), 'k+', 'Linewidth', 2, 'MarkerSize', 7);

plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);






% =========================================================================



hold off;

end
