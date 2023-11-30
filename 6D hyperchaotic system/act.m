%-------------------------------------------------------------------------%
function [act, actd, actdd, actddd] = act(x,w,b,type_Activation)
%-------------------------------------------------------------------------%

%{
Decription:
This function determines the function outputs along with its first
derivative, second derivative, and third derivative for an activation
function. So far we only are using the sigmoid function. In the future we
will add more activation functions.

Author(s):
Kristofer Drozd
Enrico Schiassi
Mario De Florio

Last Update:
02/12/2020

Inputs:
x - independent variable
w - input weight
b - bias
    
Outputs:
act - function output
actdd - first derivative output
actdd - second derivative output
actddd - third derivative output
%}

switch type_Activation
    
    case 1 % logistic
        
        act = 1/(exp(- b - w*x) + 1);
        
        
        actd = (w*exp(- b - w*x))/(exp(- b - w*x) + 1)^2;
        
        
        actdd = (2*w^2*exp(- 2*b - 2*w*x))/(exp(- b - w*x) + 1)^3 - (w^2*exp(- b - w*x))/(exp(- b - w*x) + 1)^2;
        
        
        actddd = (w^3*exp(- b - w*x))/(exp(- b - w*x) + 1)^2 - (6*w^3*exp(- 2*b - 2*w*x))/(exp(- b - w*x) + 1)^3 + (6*w^3*exp(- b - w*x)*exp(- 2*b - 2*w*x))/(exp(- b - w*x) + 1)^4 ;
        
    case 2 % tanH
        
        
        act = (exp(b + w*x) - exp(- b - w*x))/(exp(b + w*x) + exp(- b - w*x));
        
        
        actd =(w*exp(b + w*x) + w*exp(- b - w*x))/(exp(b + w*x) + exp(- b - w*x)) - ((exp(b + w*x) - exp(- b - w*x))*(w*exp(b + w*x) - w*exp(- b - w*x)))/(exp(b + w*x) + exp(- b - w*x))^2;
        
        
        actdd = (2*(exp(b + w*x) - exp(- b - w*x))*(w*exp(b + w*x) - w*exp(- b - w*x))^2)/(exp(b + w*x) + exp(- b - w*x))^3 - (2*(w*exp(b + w*x) + w*exp(- b - w*x))*(w*exp(b + w*x) - w*exp(- b - w*x)))/(exp(b + w*x) + exp(- b - w*x))^2 - (w^2*exp(- b - w*x) - w^2*exp(b + w*x))/(exp(b + w*x) + exp(- b - w*x)) - ((exp(b + w*x) - exp(- b - w*x))*(w^2*exp(- b - w*x) + w^2*exp(b + w*x)))/(exp(b + w*x) + exp(- b - w*x))^2;
        
        
        actddd = (w^3*exp(- b - w*x) + w^3*exp(b + w*x))/(exp(b + w*x) + exp(- b - w*x)) - (6*(exp(b + w*x) - exp(- b - w*x))*(w*exp(b + w*x) - w*exp(- b - w*x))^3)/(exp(b + w*x) + exp(- b - w*x))^4 + (6*(w*exp(b + w*x) + w*exp(- b - w*x))*(w*exp(b + w*x) - w*exp(- b - w*x))^2)/(exp(b + w*x) + exp(- b - w*x))^3 + ((exp(b + w*x) - exp(- b - w*x))*(w^3*exp(- b - w*x) - w^3*exp(b + w*x)))/(exp(b + w*x) + exp(- b - w*x))^2 - (3*(w^2*exp(- b - w*x) + w^2*exp(b + w*x))*(w*exp(b + w*x) + w*exp(- b - w*x)))/(exp(b + w*x) + exp(- b - w*x))^2 + (3*(w^2*exp(- b - w*x) - w^2*exp(b + w*x))*(w*exp(b + w*x) - w*exp(- b - w*x)))/(exp(b + w*x) + exp(- b - w*x))^2 + (6*(exp(b + w*x) - exp(- b - w*x))*(w^2*exp(- b - w*x) + w^2*exp(b + w*x))*(w*exp(b + w*x) - w*exp(- b - w*x)))/(exp(b + w*x) + exp(- b - w*x))^3 ;
        
    case 3 % sin
        
        
        act = sin(b + w*x);
        
        
        actd = w*cos(b + w*x);
        
        
        actdd = -w^2*sin(b + w*x);
        
        
        actddd =  -w^3*cos(b + w*x) ;
        
    case 4 % cos
        
        
        act = cos(b + w*x);
        
        
        actd = -w*sin(b + w*x);
        
        
        actdd = -w^2*cos(b + w*x);
        
        
        actddd = w^3*sin(b + w*x) ;
        
    case 5 % Gaussian
        
        
        act = exp(-(b + w*x)^2);
        
        
        actd = -2*w*exp(-(b + w*x)^2)*(b + w*x);
        
        
        actdd = 4*w^2*exp(-(b + w*x)^2)*(b + w*x)^2 - 2*w^2*exp(-(b + w*x)^2);
        
        
        actddd = 12*w^3*exp(-(b + w*x)^2)*(b + w*x) - 8*w^3*exp(-(b + w*x)^2)*(b + w*x)^3;
        
        
    case 6 % ArcTan
        
        
        act = atan(b + w*x);
        
        
        actd = w/((b + w*x)^2 + 1);
        
        
        actdd = -(2*w^2*(b + w*x))/((b + w*x)^2 + 1)^2;
        
        
        actddd =  (8*w^3*(b + w*x)^2)/((b + w*x)^2 + 1)^3 - (2*w^3)/((b + w*x)^2 + 1)^2 ;
        
    case 7 % Hyp Sine
        
        
        act = sinh(b + w*x);
        
        
        actd = w*cosh(b + w*x);
        
        
        actdd = w^2*sinh(b + w*x);
        
        
        actddd = w^3*cosh(b + w*x)  ;
        
    case 8 % SoftPlus
        
        
        act = log(exp(b + w*x) + 1);
        
        
        actd = (w*exp(b + w*x))/(exp(b + w*x) + 1);
        
        
        actdd = (w^2*exp(b + w*x))/(exp(b + w*x) + 1) - (w^2*exp(2*b + 2*w*x))/(exp(b + w*x) + 1)^2;
        
        
        actddd =  (w^3*exp(b + w*x))/(exp(b + w*x) + 1) - (3*w^3*exp(2*b + 2*w*x))/(exp(b + w*x) + 1)^2 + (2*w^3*exp(b + w*x)*exp(2*b + 2*w*x))/(exp(b + w*x) + 1)^3 ;
        
    case 9 % Bent Identity
        
        
        act = b + ((b + w*x)^2 + 1)^(1/2)/2 + w*x - 1/2;
        
        
        actd = w + (w*(b + w*x))/(2*((b + w*x)^2 + 1)^(1/2));
        
        
        actdd = w^2/(2*((b + w*x)^2 + 1)^(1/2)) - (w^2*(b + w*x)^2)/(2*((b + w*x)^2 + 1)^(3/2));
        
        
        actddd = (3*w^3*(b + w*x)^3)/(2*((b + w*x)^2 + 1)^(5/2)) - (3*w^3*(b + w*x))/(2*((b + w*x)^2 + 1)^(3/2))  ;
        
    case 10 % Inverse Hyp Sine
        
        
        act = log(b + ((b + w*x)^2 + 1)^(1/2) + w*x);
        
        
        actd = (w + (w*(b + w*x))/((b + w*x)^2 + 1)^(1/2))/(b + ((b + w*x)^2 + 1)^(1/2) + w*x);
        
        
        actdd = (w^2/((b + w*x)^2 + 1)^(1/2) - (w^2*(b + w*x)^2)/((b + w*x)^2 + 1)^(3/2))/(b + ((b + w*x)^2 + 1)^(1/2) + w*x) - (w + (w*(b + w*x))/((b + w*x)^2 + 1)^(1/2))^2/(b + ((b + w*x)^2 + 1)^(1/2) + w*x)^2;
        
        
        actddd =   (2*(w + (w*(b + w*x))/((b + w*x)^2 + 1)^(1/2))^3)/(b + ((b + w*x)^2 + 1)^(1/2) + w*x)^3 - ((3*w^3*(b + w*x))/((b + w*x)^2 + 1)^(3/2) - (3*w^3*(b + w*x)^3)/((b + w*x)^2 + 1)^(5/2))/(b + ((b + w*x)^2 + 1)^(1/2) + w*x) - (3*(w^2/((b + w*x)^2 + 1)^(1/2) - (w^2*(b + w*x)^2)/((b + w*x)^2 + 1)^(3/2))*(w + (w*(b + w*x))/((b + w*x)^2 + 1)^(1/2)))/(b + ((b + w*x)^2 + 1)^(1/2) + w*x)^2;
        
    case 11 % Softsign
        
        
        act = (b + w*x)/(abs(b + w*x) + 1);
        
        
        actd = w/(abs(b + w*x) + 1) - (w*sign(b + w*x)*(b + w*x))/(abs(b + w*x) + 1)^2;
        
        
        actdd =(2*w^2*sign(b + w*x)^2*(b + w*x))/(abs(b + w*x) + 1)^3 - (2*w^2*sign(b + w*x))/(abs(b + w*x) + 1)^2 - (2*w^2*dirac(b + w*x)*(b + w*x))/(abs(b + w*x) + 1)^2;
        
        
        actddd = (6*w^3*sign(b + w*x)^2)/(abs(b + w*x) + 1)^3 - (6*w^3*dirac(b + w*x))/(abs(b + w*x) + 1)^2 - (6*w^3*sign(b + w*x)^3*(b + w*x))/(abs(b + w*x) + 1)^4 - (2*w^3*(b + w*x)*dirac(1, b + w*x))/(abs(b + w*x) + 1)^2 + (12*w^3*dirac(b + w*x)*sign(b + w*x)*(b + w*x))/(abs(b + w*x) + 1)^3  ;
        
        
end

end