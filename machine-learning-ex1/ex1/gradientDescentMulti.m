function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    cols=size(X,2);
    sum=zeros(1,cols);
    for i=1:m
        hx=theta'*X(i,:)';
        for j=1:cols
            sum(j)=sum(j)+(hx-y(i,1))*X(i,j);  
        end
    end
    
    for j=1:cols
    theta(j,1)=theta(j,1)-(alpha/m)*sum(j);
    end
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
