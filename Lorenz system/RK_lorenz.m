clear; close all; clc;
format long
%--------------------------------------------------------------------------
%{ 
  Runge-Kutta for data generation
  Test Case - Lorenz System

  Author:
  Mario De Florio
%}

% create data directory
mkdir data

% Parameters
sigma = 10;
rho = 27;
beta = 2.6667;

% Define the time span and step size
tspan = [0 10];
h_step = 0.01; % Step size
t = tspan(1):h_step:tspan(2);

% Initial conditions
x0 = -8;
y0 = 7;
z0 = 27;

% Function for the Lorenz system
f = @(t, y) [sigma * (y(2) - y(1)); 
            y(1) * (rho - y(3)) - y(2);  
            y(1) * y(2) - beta * y(3)];

% Initialize arrays for the solution
x = zeros(size(t));
y = zeros(size(t));
z = zeros(size(t));

% Set initial values
x(1) = x0;
y(1) = y0;
z(1) = z0;

% Implementing Runge-Kutta method
for i = 1:length(t)-1

    k1 = h_step * f(t(i), [x(i), y(i), z(i)]);
    k2 = h_step * f(t(i) + h_step/2, [x(i) + k1(1)/2, y(i) + k1(2)/2, z(i) + k1(3)/2]);
    k3 = h_step * f(t(i) + h_step/2, [x(i) + k2(1)/2, y(i) + k2(2)/2, z(i) + k2(3)/2]);
    k4 = h_step * f(t(i) + h_step, [x(i) + k3(1), y(i) + k3(2), z(i) + k3(3)]);
    
    x(i+1) = x(i) + (1/6) * (k1(1) + 2*k2(1) + 2*k3(1) + k4(1));
    y(i+1) = y(i) + (1/6) * (k1(2) + 2*k2(2) + 2*k3(2) + k4(2));
    z(i+1) = z(i) + (1/6) * (k1(3) + 2*k2(3) + 2*k3(3) + k4(3));

end

rhs_1_anal = sigma * (y - x);
rhs_2_anal = x .* (rho - z) - y;
rhs_3_anal = x .* y - beta * z;

y1_anal = x;
y2_anal = y;
y3_anal = z;

% save data
directory = 'data';

% Save the variables in the specified file
save(fullfile(directory, 'data_generated.mat'), 'y1_anal', 'y2_anal', 'y3_anal', 'rhs_1_anal', 'rhs_2_anal', 'rhs_3_anal', 't', 'h_step');


% Plotting the solutions


figure(1)
subplot(3,1,1)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,y1_anal,'LineWidth',2)
box on
title('y1', 'FontWeight', 'Normal')

subplot(3,1,2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,y2_anal,'LineWidth',2)
box on
title('y2', 'FontWeight', 'Normal')

subplot(3,1,3)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,y3_anal,'LineWidth',2)
box on
title('y3', 'FontWeight', 'Normal')


