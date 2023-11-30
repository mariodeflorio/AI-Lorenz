clear; close all; clc;
format long
%--------------------------------------------------------------------------
%{ 
  Runge-Kutta to test RK generated data vs. PySR discovered equations 
  Test Case - 6D Hyperchaotis System

  Author:
  Mario De Florio
%}

%% =======================================

%% Build the system discovered by PySR
f = @(t, y) [-10.00001*y(1) + 10.00001*y(2) + y(4); 
              y(1)*(27.99999 - y(3)) - y(2) - y(5) ; 
            y(1)*y(2) - 2.6667*y(3) ;
             - y(2)*y(3) - y(4); 
             3*y(2) ; 
            y(3)*y(4) - 10*y(6) ];

% synthetic data (exact solution)
% Specify the directory
directory = 'data';

% Load the variables from the specified file
load(fullfile(directory, 'data_generated.mat'));
load(fullfile(directory, 'bbxtfc_data.mat'));



% Runge-Kutta for 3D Hyperchaotic system
% Define the time span and step size
t_0 = t(1);
t_f = t(end);
tspan = [t_0 t_f];
t_obs = linspace(t_0,t_f,length(y1_data_pert));
t = tspan(1):h_step:tspan(2);

% Initial conditions
x0 = .1;
y0 = .1;
z0 = .1;
u0 = .1;
v0 = .1;
w0 = .1;

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



sol1 = x;
sol2 = y;
sol3 = z;
sol4 = u;
sol5 = v;
sol6 = w;


% Plotting the solutions

figure(1)
subplot(6,3,[1,4,7])
set(gca,'Fontsize',12)
hold on
plot3(y1_data_pert,y2_data_pert,y3_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot3(y1_anal,y2_anal,y3_anal,'LineWidth',2, 'Color','#15607a')
plot3(sol1,sol2,sol3,':','LineWidth',2, 'Color','#ff483a')
view([-45 45 15])
axis off

subplot(6,3,[10,13,16])
set(gca,'Fontsize',12)
hold on
plot3(y4_data_pert,y5_data_pert,y6_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot3(y4_anal,y5_anal,y6_anal,'LineWidth',2, 'Color','#15607a')
plot3(sol4,sol5,sol6,':','LineWidth',2, 'Color','#ff483a')
view([-45 45 15])
axis off

l_w = 2; % line width


subplot(6,3,2)
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

subplot(6,3,5)
set(gca,'Fontsize',12)
hold on
box on
plot(t_obs,y2_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot(t,y2_anal,'LineWidth',l_w, 'Color','#15607a')
plot(t,sol2,':','LineWidth',l_w, 'Color',  '#ff483a')
ylabel('y','Rotation',0)
xlim([t_0 t_f])
set(gca,'XTick',[]); % Hide x-axis ticks
set(gca,'YTick',[]); % Hide y-axis ticks


subplot(6,3,8)
set(gca,'Fontsize',12)
hold on
box on
plot(t_obs,y3_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot(t,y3_anal,'LineWidth',l_w, 'Color','#15607a')
plot(t,sol3,':','LineWidth',l_w, 'Color',  '#ff483a')
ylabel('z','Rotation',0)
xlim([t_0 t_f])
set(gca,'YTick',[]); % Hide y-axis ticks
set(gca,'XTick',[]); % Hide x-axis ticks


subplot(6,3,11)
set(gca,'Fontsize',12)
hold on
box on
plot(t_obs,y4_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot(t,y4_anal,'LineWidth',l_w, 'Color','#15607a')
plot(t,sol4,':','LineWidth',l_w, 'Color',  '#ff483a')
ylabel('u','Rotation',0)
xlim([t_0 t_f])
set(gca,'XTick',[]); % Hide x-axis ticks
set(gca,'YTick',[]); % Hide y-axis ticks

subplot(6,3,14)
set(gca,'Fontsize',12)
hold on
box on
plot(t_obs,y5_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot(t,y5_anal,'LineWidth',l_w, 'Color','#15607a')
plot(t,sol5,':','LineWidth',l_w, 'Color',  '#ff483a')
ylabel('v','Rotation',0)
xlim([t_0 t_f])
set(gca,'XTick',[]); % Hide x-axis ticks
set(gca,'YTick',[]); % Hide y-axis ticks


subplot(6,3,17)
set(gca,'Fontsize',12)
hold on
box on
plot(t_obs,y6_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
plot(t,y6_anal,'LineWidth',l_w, 'Color','#15607a')
plot(t,sol6,':','LineWidth',l_w, 'Color',  '#ff483a')
xlabel('t')
ylabel('w','Rotation',0)
xlim([t_0 t_f])
set(gca,'YTick',[]); % Hide y-axis ticks

