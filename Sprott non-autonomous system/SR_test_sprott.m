clear; close all; clc;
format long
%--------------------------------------------------------------------------
%{ 
  Runge-Kutta to test RK generated data vs. PySR discovered equations 
  Test Case - Lorenz System

  Author:
  Mario De Florio
%}

%% =======================================

%% Build the system discovered by PySR
f = @(t, y) [3*sin(y(2))*y(3) + sin(6.2831726*t); 
             sin(y(1)) - sin(y(2));  
             0.333 - 3.0000002*sin(y(1))^2];



% synthetic data (exact solution)

% Specify the directory
directory = 'data';

% Load the variables from the specified file
load(fullfile(directory, 'data_generated.mat'));
load(fullfile(directory, 'bbxtfc_data.mat'));


% Runge-Kutta for Lorenz system
% Define the time span and step size
t_0 = 0;
t_f = t(end);
tspan = [t_0 t_f];
t_obs = linspace(t_0,t_f,length(y1_data_pert));
t = tspan(1):h_step:tspan(2);

% Initial conditions
x0 = 0.01 + 2*pi;
y0 = 0.1 + 2*pi;
z0 = 0.1;

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

sol1 = x;
sol2 = y;
sol3 = z;

% Plotting the solutions

figure(1)
subplot(3,2,[1,3,5])
set(gca,'Fontsize',12)
hold on

plot3(y1_data_pert,y2_data_pert,y3_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot3(y1_anal,y2_anal,y3_anal,'LineWidth',2, 'Color','#15607a')
plot3(sol1,sol2,sol3,':','LineWidth',2, 'Color','#ff483a')
view([-45 45 15])
axis off

l_w = 2; % line width

subplot(3,2,2)
set(gca,'Fontsize',12)
hold on
box on
plot(t_obs,y1_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot(t,y1_anal,'LineWidth',l_w, 'Color','#15607a')
plot(t,sol1,':','LineWidth',l_w, 'Color',  '#ff483a')
ylabel('x','Rotation',0)
legend(['Data points with noise \sigma=', num2str(noise_std)],'Exact Dynamics','Discovered Dynamics')
xlim([t_0 t_f])
set(gca,'XTick',[]); % Hide x-axis ticks
set(gca,'YTick',[]); % Hide y-axis ticks

subplot(3,2,4)
set(gca,'Fontsize',12)
hold on
box on
plot(t_obs,y2_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot(t,y2_anal,'LineWidth',l_w,'Color', '#15607a')
plot(t,sol2,':','LineWidth',l_w, 'Color',  '#ff483a')
ylabel('y','Rotation',0)
xlim([t_0 t_f])
set(gca,'XTick',[]); % Hide x-axis ticks
set(gca,'YTick',[]); % Hide y-axis ticks


subplot(3,2,6)
set(gca,'Fontsize',12)
hold on
box on
plot(t_obs,y3_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot(t,y3_anal,'LineWidth',l_w,'Color', '#15607a')
plot(t,sol3,':','LineWidth',l_w, 'Color',  '#ff483a')
xlabel('t')
ylabel('z','Rotation',0)
xlim([t_0 t_f])
set(gca,'YTick',[]); % Hide y-axis ticks

