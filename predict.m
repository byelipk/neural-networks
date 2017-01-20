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

a1 = [ ones(m, 1), X ];

% z2 is the linear combination of our weights and the input values. To
% compute this using matrix multiplication we need to take the transpose of
% a1 (so the number of columns in Theta1 equals the number of rows in a1).
% z2 is now a (4 x 8) matrix.
z2 = Theta1 * a1';

% Compute values for activation layer 2 by applying z2 to the sigmoid function.
% We also set z2 to the transpose of itself to make multiplication easier.
z2 = sigmoid(z2);
z2 = z2';

% Next we add the bias unit by adding a column of 1's.
a2 = [ones(size(z2, 1), 1), z2];

% Multiply Theta2 by the transpose of a2.
z3 = Theta2 * a2';

% The activation of the only unit in the output layer.
a3 = sigmoid(z3);
a3 = a3';

[a, p] = max(a3, [], 2);







% =========================================================================


end
