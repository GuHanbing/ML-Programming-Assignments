function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

rows=size(X,1);
cols=size(X,2);
hx=sigmoid(X*theta);     %rows*1µÄh_theta(x^i)µÄÖµ
for i=1:rows
    J=J-1/m*(y(i)*log(hx(i))+(1-y(i))*log(1-hx(i)));
    for j=1:cols
    grad(j)=grad(j)+1/m*(hx(i)-y(i))*X(i,j);
    end
end

exJ=lambda/(2*m)*(theta'*theta-theta(1)*theta(1))*ones(size(J));
exGrad=lambda/m*theta;
exGrad(1)=0;
J=J+exJ;
grad=grad+exGrad;



% =============================================================

end
