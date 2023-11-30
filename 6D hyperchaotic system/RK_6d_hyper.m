clear; close all; clc;
format long

%--------------------------------------------------------------------------
%{ 
  Runge-Kutta for data generation
  Test Case - 6D Hyperchaotic System

  Author:
  Mario De Florio
%}

mkdir data


% Parameters
a = 10;
b = 2.6667;
c = 28;
d = -1;
e = 10;
r = 3 ;

% Define the time span and step size
tspan = [0 10];
h_step = 0.01; % Step size
t = tspan(1):h_step:tspan(2);

% Initial conditions
x0 = .1;
y0 = .1;
z0 = .1;
u0 = .1;
v0 = .1;
w0 = .1;

% Function for the Lorenz system
f = @(t, y) [a*(y(2) - y(1)) + y(4); 
             c*y(1) - y(2) - y(1)*y(3) - y(5); 
             y(1)*y(2) - b*y(3) ;
             d*y(4) - y(2)*y(3); 
             r*y(2) ; 
             -e*y(6) + y(3)*y(4) ];

% Initialize arrays for the solution
x = zeros(size(t));
y = zeros(size(t));
z = zeros(size(t));
u = zeros(size(t));
v = zeros(size(t));
w = zeros(size(t));

% Set initial values
x(1) = x0;
y(1) = y0;
z(1) = z0;
u(1) = u0;
v(1) = v0;
w(1) = w0;

% Implementing Runge-Kutta method
for i = 1:length(t)-1
    k1 = h_step * f(t(i), [x(i), y(i), z(i), u(i), v(i), w(i)]);
    k2 = h_step * f(t(i) + h_step/2, [x(i) + k1(1)/2, y(i) + k1(2)/2, z(i) + k1(3)/2, u(i) + k1(4)/2, v(i) + k1(5)/2, w(i) + k1(6)/2 ]);
    k3 = h_step * f(t(i) + h_step/2, [x(i) + k2(1)/2, y(i) + k2(2)/2, z(i) + k2(3)/2, u(i) + k2(4)/2, v(i) + k2(5)/2, w(i) + k2(6)/2 ]);
    k4 = h_step * f(t(i) + h_step, [x(i) + k3(1), y(i) + k3(2), z(i) + k3(3), u(i) + k3(4), v(i) + k3(5), w(i) + k3(6)]);
    
    x(i+1) = x(i) + (1/6) * (k1(1) + 2*k2(1) + 2*k3(1) + k4(1));
    y(i+1) = y(i) + (1/6) * (k1(2) + 2*k2(2) + 2*k3(2) + k4(2));
    z(i+1) = z(i) + (1/6) * (k1(3) + 2*k2(3) + 2*k3(3) + k4(3));
    u(i+1) = u(i) + (1/6) * (k1(4) + 2*k2(4) + 2*k3(4) + k4(4));
    v(i+1) = v(i) + (1/6) * (k1(5) + 2*k2(5) + 2*k3(5) + k4(5));
    w(i+1) = w(i) + (1/6) * (k1(6) + 2*k2(6) + 2*k3(6) + k4(6));
end

rhs_1_anal = a*(y - x) + u; 
rhs_2_anal = c*x - x.*z - y - v; 
rhs_3_anal = x.*y - b*z ;
rhs_4_anal = d*u - y.*z; 
rhs_5_anal = r*y ; 
rhs_6_anal = -e*w + z.*u;

y1_anal = x;
y2_anal = y;
y3_anal = z;
y4_anal = u;
y5_anal = v;
y6_anal = w;

% save data

directory = 'data';

% Save the variables in the specified file
save(fullfile(directory, 'data_generated.mat'), 'y1_anal', 'y2_anal', 'y3_anal', 'y4_anal', 'y5_anal', 'y6_anal', ...
    'rhs_1_anal', 'rhs_2_anal', 'rhs_3_anal', 'rhs_4_anal', 'rhs_5_anal', 'rhs_6_anal', 't', 'h_step');


% Plotting the solutions

figure(1)
subplot(6,2,1)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,y1_anal,'LineWidth',2)
box on

subplot(6,2,3)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,y2_anal,'LineWidth',2)
box on

subplot(6,2,5)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,y3_anal,'LineWidth',2)
box on

subplot(6,2,7)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,y4_anal,'LineWidth',2)
box on

subplot(6,2,9)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,y5_anal,'LineWidth',2)
box on

subplot(6,2,11)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,y6_anal,'LineWidth',2)
box on


subplot(6,2,2)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,rhs_1_anal,'LineWidth',2)
box on

subplot(6,2,4)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,rhs_2_anal,'LineWidth',2)
box on

subplot(6,2,6)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,rhs_3_anal,'LineWidth',2)
box on

subplot(6,2,8)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,rhs_4_anal,'LineWidth',2)
box on

subplot(6,2,10)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,rhs_5_anal,'LineWidth',2)
box on

subplot(6,2,12)
set(gca,'Fontsize',12)
hold on
grid on 
plot(t,rhs_6_anal,'LineWidth',2)
box on

