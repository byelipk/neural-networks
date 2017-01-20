Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
X = reshape(sin(1:16), 8, 2);


% FORWARD PROPAGATION

% Add the bias unit and rename to 'a1' - we can think of this as representing
% the activation of our input layer.
a1 = [ ones(size(X, 1), 1), X ];

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



% p = predict(Theta1, Theta2, X)
% you should see this result
% p =
%   4
%   1
%   1
%   4
%   4
%   4
%   4
%   2
