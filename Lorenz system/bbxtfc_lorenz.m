clear; close all; clc;
format long
%--------------------------------------------------------------------------
%{ 
  Black-Box X-TFC applied to Chaotic Systems
  Test Case - Lorenz System

  Author:
  Mario De Florio
%}

rng('default') % set random seed

%% =======================================
% synthetic data from Runge-Kutta 

% Specify the directory
directory = 'data';
% Load the variables from the specified file
load(fullfile(directory, 'data_generated.mat'));

%--------------------------------------------------------------------------
%% Input

% Define the standard deviation of the Gaussian noise
noise_std = 0; % Adjust this as needed [ 0 , 0.1 , 0.2 , 1.0 , 2.0 ]

if noise_std == 0
    N = 20;    % Number of collocation points in each subdomain
    m = 20;    % number of neurons
    t_step = 0.1; % length of each subdomain
else
    N = 100;    
    m = 100;    
    t_step = 1;
end

D = 3; % dimensions of the system
start = tic;

t_0 = 0; % initial time
t_f = t(end); % final time

x = linspace(-1,1,N)'; % Discretization of x (-1,1) 

t_tot = (t_0:t_step:t_f)';
n_t = length(t_tot);

n_points = n_t + (n_t-1)*(N-2); 
t_domain = linspace(t_0,t_f,n_points);

LB = -1; % Lower boundary for weight and bias samplings
UB = 1; % Upper boundary for weight and bias samplings

% iterative least-square parameters
IterMax = 100;
IterTol = 1e-16;

type_act = 2; % activation function
%{
1= Logistic;
2= TanH;
3= Sine;
4= Cosine;
5= Gaussian; the best so far w/ m=11
6= ArcTan;
7= Hyperbolic Sine;
8= SoftPlus
9= Bent Identity;
10= Inverse Hyperbolic Sine
11= Softsign
%}

t_obs = linspace(t_0,t_f,length(y1_anal));

%% Data Perturbation
noise = noise_std * randn(size(y1_anal));

y1_data_pert = y1_anal + noise;
y2_data_pert = y2_anal + noise;
y3_data_pert = y3_anal + noise;

%% interpolation observed data

y_RK_inter = 1:length(y1_data_pert);
ind = linspace(1,length(y1_data_pert),n_points);

y1_data = spline(y_RK_inter,y1_data_pert,ind)';
y2_data = spline(y_RK_inter,y2_data_pert,ind)';
y3_data = spline(y_RK_inter,y3_data_pert,ind)';

rhs_1_data = spline(y_RK_inter,rhs_1_anal,ind)';
rhs_2_data = spline(y_RK_inter,rhs_2_anal,ind)';
rhs_3_data = spline(y_RK_inter,rhs_3_anal,ind)';


%% Activation function definition

weight = unifrnd(LB,UB,m,1);
bias = unifrnd(LB,UB,m,1);

h= zeros(N,m); hd= zeros(N,m); hdd= zeros(N,m);

for i = 1 : N
    for j = 1 : (m)
        [h(i, j), hd(i, j), hdd(i,j)] = act(x(i),weight(j), bias(j),type_act);
    end
end

h0 = h(1,:);

%% BBX-TFC construction

% Initial Values
y1_0 = y1_data(1);
y2_0 = y2_data(1);
y3_0 = y3_data(1);

sol1 = zeros(n_t + (n_t-1)*(N-2),1);
sol2 = zeros(n_t + (n_t-1)*(N-2),1);
sol3 = zeros(n_t + (n_t-1)*(N-2),1);

sol1_dot = zeros(n_points,1); 
sol2_dot = zeros(n_points,1); 
sol3_dot = zeros(n_points,1); 

training_err_vec = zeros(n_t-1,1);

%  Jacobian matrix
JJ = zeros(D*N, D*m);
for ii = 1:D
    JJ((ii - 1) * N + 1 : ii * N, (ii - 1) * m + 1 : ii * m) = - (h-h0);
end

tStart = tic;

%% Computation over each subdomain
for i = 1:(n_t-1)

    xi_1 = zeros(m,1);
    xi_2 = zeros(m,1);
    xi_3 = zeros(m,1);
     
    y1_data_i = y1_data((N-1)*(i-1)+1:(N-1)*i+1) ;
    y2_data_i = y2_data((N-1)*(i-1)+1:(N-1)*i+1) ;
    y3_data_i = y3_data((N-1)*(i-1)+1:(N-1)*i+1) ;

    c_i = (x(end) - x(1)) / (t_tot(i+1) - t_tot(i));
    
    xi = [xi_1;xi_2;xi_3];

    %% Build Constrained Expressions
    
    y1 = (h-h0)*xi_1 + y1_0;        y1_dot = c_i*hd*xi_1;     
    y2 = (h-h0)*xi_2 + y2_0;        y2_dot = c_i*hd*xi_2;   
    y3 = (h-h0)*xi_3 + y3_0;        y3_dot = c_i*hd*xi_3;
   
    %% Build the Losses  

    L_data_1 = y1_data_i - y1 ;
    L_data_2 = y2_data_i - y2 ;
    L_data_3 = y3_data_i - y3 ;

    Loss = [ L_data_1  ; L_data_2 ; L_data_3  ];
    
    % X-TFC ILS loop
    l2 = [2 1];
    iter = 0;
    
    while abs(l2(2)) > IterTol &&  iter < IterMax && abs(l2(1) - l2(2)) > IterTol
        
        l2(1)= l2(2);
        
        % xi variation
        dxi = lsqminnorm(JJ, Loss);
        
        % update xi
        xi = xi - dxi;
        
        xi_1 = xi((0*m)+1:1*m);
        xi_2 = xi((1*m)+1:2*m);
        xi_3 = xi((2*m)+1:3*m);   
    
        %% Re-Build Constrained Expressions
        
        y1 = (h-h0)*xi_1 + y1_0;        y1_dot = c_i*hd*xi_1;
        y2 = (h-h0)*xi_2 + y2_0;        y2_dot = c_i*hd*xi_2;
        y3 = (h-h0)*xi_3 + y3_0;        y3_dot = c_i*hd*xi_3;
        
        %% Re-Build the Losses
        
        L_data_1 = y1_data_i - y1  ;
        L_data_2 = y2_data_i - y2  ;
        L_data_3 = y3_data_i - y3  ;

        Loss =  [ L_data_1  ; L_data_2  ; L_data_3 ];

        l2(2) = norm(Loss);
        iter = iter+1;
        
    end
    
    training_err = sqrt(mean(abs(L_data_1.^2))) + sqrt(mean(abs(L_data_2.^2))) +  sqrt(mean(abs(L_data_3.^2)))  ;  

    % Update of constraints
    y1_0 = y1(end);
    y2_0 = y2(end);
    y3_0 = y3(end);

	sol1((N-1)*(i-1)+1:(N-1)*i+1) = y1;
    sol2((N-1)*(i-1)+1:(N-1)*i+1) = y2;
    sol3((N-1)*(i-1)+1:(N-1)*i+1) = y3;

    sol1_dot((N-1)*(i-1)+1:(N-1)*i+1) = y1_dot;
    sol2_dot((N-1)*(i-1)+1:(N-1)*i+1) = y2_dot;
    sol3_dot((N-1)*(i-1)+1:(N-1)*i+1) = y3_dot;

    training_err_vec(i) = training_err;
             
end

xtfc_elapsedtime = toc(tStart) ;

%%

% Smoothing the outliers

if ismember(noise_std, [0.1, 0.5])
    window_size = 31;  % You can adjust this to control the amount of smoothing
elseif ismember(noise_std, [1.0, 2.0])
    window_size = 55;
else
    % Handle other cases if needed
end

polynomial_order = 5;  % Order of the polynomial to fit

if noise_std ~= 0
    sol1_dot = sgolayfilt(sol1_dot, polynomial_order, window_size);
    sol2_dot = sgolayfilt(sol2_dot, polynomial_order, window_size);
    sol3_dot = sgolayfilt(sol3_dot, polynomial_order, window_size);
    sol1 = sgolayfilt(sol1, polynomial_order, window_size);
    sol2 = sgolayfilt(sol2, polynomial_order, window_size);
    sol3 = sgolayfilt(sol3, polynomial_order, window_size);
end

fprintf('The elapsed time for x-tfc is: %g \n', xtfc_elapsedtime );
fprintf('\n')
fprintf('The average training error for X-TFC is: %g \n', mean(training_err_vec) )


% save data
data_for_pysr = [t_domain',  sol1, sol2, sol3, sol1_dot, sol2_dot, sol3_dot];
writematrix(data_for_pysr, fullfile(directory, 'pysr_data.csv'));
% Save the variables in the specified file
save(fullfile(directory, 'bbxtfc_data.mat'), 'y1_data_pert', 'y2_data_pert', 'y3_data_pert', 'noise_std');

%% Errors

MAE_dyn_1 = mean(abs(y1_data - sol1));
MAE_dyn_2 = mean(abs(y2_data - sol2));
MAE_dyn_3 = mean(abs(y3_data - sol3));

MAE_rhs_1 = mean(abs(rhs_1_data - sol1_dot));
MAE_rhs_2 = mean(abs(rhs_2_data - sol2_dot));
MAE_rhs_3 = mean(abs(rhs_3_data - sol3_dot));

fprintf('\n')
fprintf('MAE of x: %.2e\n', MAE_dyn_1)
fprintf('MAE of y: %.2e\n', MAE_dyn_2)
fprintf('MAE of z: %.2e\n', MAE_dyn_3)
fprintf('\n')
fprintf('MAE of f: %.2e\n', MAE_rhs_1)
fprintf('MAE of h: %.2e\n', MAE_rhs_2)
fprintf('MAE of k: %.2e\n', MAE_rhs_3)

%% Plots

if noise_std == 0

    figure(1)
    subplot(3,4,[1,5,9])
    set(gca,'Fontsize',12)
    hold on
    plot3(y1_anal,y2_anal,y3_anal,'LineWidth',2, 'Color','#15607a')
    plot3(sol1,sol2,sol3,'--','LineWidth',2, 'Color','#65e0ba')
    view([-45 45 15])
    legend('Exact','Learned','Position',[0.785 0.14 0.08 0.1])
    axis off

    subplot(3,4,2)
    set(gca,'Fontsize',12)
    hold on
    box on
    plot(t_obs,y1_anal,'LineWidth',2, 'Color','#15607a')
    plot(t_domain,sol1,'--','LineWidth',2, 'Color',  '#65e0ba')
    ylabel('x','Rotation',0)
    xlim([t_0 t_f])
    set(gca,'XTick',[]); % Hide x-axis ticks
    set(gca,'YTick',[]); % Hide y-axis ticks

    subplot(3,4,6)
    set(gca,'Fontsize',12)
    hold on
    box on
    plot(t_obs,y2_anal,'LineWidth',2,'Color', '#15607a')
    plot(t_domain,sol2,'--','LineWidth',2, 'Color',  '#65e0ba')
    ylabel('y','Rotation',0)
    xlim([t_0 t_f])
    set(gca,'XTick',[]); % Hide x-axis ticks
    set(gca,'YTick',[]); % Hide y-axis ticks

    subplot(3,4,10)
    set(gca,'Fontsize',12)
    hold on
    box on
    plot(t_obs,y3_anal,'LineWidth',2,'Color', '#15607a')
    plot(t_domain,sol3,'--','LineWidth',2, 'Color',  '#65e0ba')
    xlabel('t')
    ylabel('z','Rotation',0)
    xlim([t_0 t_f])
    set(gca,'YTick',[]); % Hide y-axis ticks

    subplot(3,4,3)
    set(gca,'Fontsize',12)
    hold on
    box on
    plot(t_obs,rhs_1_anal,'LineWidth',2, 'Color', '#15607a')
    plot(t_domain,sol1_dot,'--','LineWidth',2, 'Color',  '#65e0ba')
    ylabel('$\mathbf{\dot{x}}$','Interpreter','latex','Rotation',0)
    xlim([t_0 t_f])
    set(gca,'XTick',[]); % Hide x-axis ticks
    set(gca,'YTick',[]); % Hide y-axis ticks

    subplot(3,4,7)
    set(gca,'Fontsize',12)
    hold on
    box on
    plot(t_obs,rhs_2_anal,'LineWidth',2, 'Color', '#15607a')
    plot(t_domain,sol2_dot,'--','LineWidth',2, 'Color',  '#65e0ba')
    ylabel('$\mathbf{\dot{y}}$','Interpreter','latex','Rotation',0)
    xlim([t_0 t_f])
    set(gca,'XTick',[]); % Hide x-axis ticks
    set(gca,'YTick',[]); % Hide y-axis ticks

    subplot(3,4,11)
    set(gca,'Fontsize',12)
    hold on
    box on
    plot(t_obs,rhs_3_anal,'LineWidth',2, 'Color', '#15607a')
    plot(t_domain,sol3_dot,'--','LineWidth',2, 'Color',  '#65e0ba')
    xlabel('t')
    ylabel('$\mathbf{\dot{z}}$','Interpreter','latex','Rotation',0)
    xlim([t_0 t_f])
    set(gca,'YTick',[]); % Hide y-axis ticks

    subplot(3,4,[4,8,12])
    set(gca,'Fontsize',12)
    hold on
    plot3(rhs_1_anal,rhs_2_anal,rhs_3_anal,'LineWidth',2, 'Color','#15607a')
    plot3(sol1_dot,sol2_dot,sol3_dot,'--','LineWidth',2, 'Color','#65e0ba')
    view([-45 45 15])
    axis off

    % Adjust the spacing between subplots
    set(gcf, 'Position',  [700, 1000, 1500, 400]); % Adjust figure size
    % Add a title for the entire figure
    annotation('textbox', [0.2, 0.9, 0.2, 0.1], 'String', 'Trajectory', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
    % Add a title for columns 3 and 4
    annotation('textbox', [0.6, 0.9, 0.2, 0.1], 'String', 'Right-Hand-Side', 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

end





if noise_std ~= 0

    figure(2)
    subplot(3,2,1)
    set(gca,'Fontsize',12)
    hold on
    plot(t_obs,y1_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
    plot(t_obs,y1_anal,'LineWidth',2, 'Color','#15607a')
    plot(t_domain,sol1,':','LineWidth',2, 'Color',  '#ff483a')
    legend(['Noisy data with \sigma=', num2str(noise_std)],'Exact Dyncamics','Learned Dynamics','Position', [0.394641238610707,0.884799021303165,0.147820160117721,0.114785989086915])
    ylabel('x','Rotation',0)
    set(gca,'XTick',[]); % Hide x-axis ticks


    subplot(3,2,3)
    set(gca,'Fontsize',12)
    hold on
    plot(t_obs,y2_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
    plot(t_obs,y2_anal,'LineWidth',2, 'Color','#15607a')
    plot(t_domain,sol2,':','LineWidth',2, 'Color',  '#ff483a')
    ylabel('y','Rotation',0)
    set(gca,'XTick',[]); % Hide x-axis ticks


    subplot(3,2,5)
    set(gca,'Fontsize',12)
    hold on
    plot(t_obs,y3_data_pert,'.','MarkerSize',10,'Color',"#7E2F8E")
    plot(t_obs,y3_anal,'LineWidth',2, 'Color','#15607a')
    plot(t_domain,sol3,':','LineWidth',2, 'Color',  '#ff483a')
    ylabel('z','Rotation',0)
    xlabel('t')


    subplot(3,2,2)
    set(gca,'Fontsize',12)
    hold on
    plot(0,0)
    plot(t_obs,rhs_1_anal,'LineWidth',2, 'Color','#15607a')
    plot(t_domain,sol1_dot,':','LineWidth',2, 'Color',  '#ff483a')
    ylabel('$\mathbf{\dot{x}}$','Interpreter','latex','Rotation',0)
    set(gca,'XTick',[]); % Hide x-axis ticks



    subplot(3,2,4)
    set(gca,'Fontsize',12)
    hold on
    plot(0,0)
    plot(t_obs,rhs_2_anal,'LineWidth',2, 'Color','#15607a')
    plot(t_domain,sol2_dot,':','LineWidth',2, 'Color',  '#ff483a')
    ylabel('$\mathbf{\dot{y}}$','Interpreter','latex','Rotation',0)
    set(gca,'XTick',[]); % Hide x-axis ticks

    subplot(3,2,6)
    set(gca,'Fontsize',12)
    hold on
    plot(0,0)
    plot(t_obs,rhs_3_anal,'LineWidth',2, 'Color','#15607a')
    plot(t_domain,sol3_dot,':','LineWidth',2, 'Color',  '#ff483a')
    ylabel('$\mathbf{\dot{z}}$','Interpreter','latex','Rotation',0)
    xlabel('t')

    set(gcf, 'Position',  [3500, 1000, 1500, 600]); % Adjust figure size
end
