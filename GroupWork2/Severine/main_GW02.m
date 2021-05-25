%**************************************************************************
%   Model Predictive Engine Control                                       *
%   Spring 2021, IDSC, ETH Zurich                                         *
%   Group Work 02                                                TEMPLATE *
%**************************************************************************

%% Initialization

% reset matlab
clear; clear functions; path(pathdef); clc; close all; %#ok<CLFUNC>

% add all needes paths to the matlab search path
pathProvidedCode = '../../providedCode';
addpath(pathProvidedCode);
if ispc == 1
    addpath([pathProvidedCode,'/casadi-matlabR2014b-v3.3.0']);
elseif ismac == 1
    addpath([pathProvidedCode,'/casadi-matlabR2015a-v3.3.0']);
else
    error('Unrecognized OS. Neither PC nor Mac!')
end % if
addpath([pathProvidedCode,'/qpOASES_Matlab']);
addpath([pathProvidedCode,'/qpOASES_Simulink']);
addpath([pathProvidedCode,'/SfunctionGeneration']);
addpath([pathProvidedCode,'/MVM']);

%% NMPC Parameters for the NLP formulation with CasADi

parNMPC.nStates  	= 5;
parNMPC.nInputs    	= 2; 
parNMPC.nOutputs  	= 2;
parNMPC.Ts         	= 0.1;         % NMPC discrete model sampling time
parNMPC.N         	= 20;          % NMPC prediction and control horizon
parNMPC.nRK4       	= 5;           % Number of RK4 intervals per time step

%% Create CasADi functions for NLP with SQP & Multiple Shooting

[qp_W,qp_gradJ,qp_gradhT,qp_h,FJk] = createCasadiFunctions(parNMPC);

%% NMPC Parameters for the Simulation

% Algorithm parameters
parNMPC.Q1_val      = 1e-6;        % Weighting on p_IM tracking error
parNMPC.Q2_val      = 1e4;         % Weighting on x_bg tracking error
parNMPC.R1_val      = 1e4;         % Weighting on change in u_vtg
parNMPC.R2_val      = 1e4;         % Weighting on change in u_egr
parNMPC.nSQP        = 1;           % Maximum SQP iterations
parNMPC.alphaSQP    = 0.3;         % Step length for damped Newton's method
parNMPC.QPMaxIter   = 60;          % Max. number of QP iterations
parNMPC.kktTol      = 1e-3;        % Stopping criteria for KKT conditions

% Number of optimisation variables and constraints
parNMPC.nOptVars    = (parNMPC.nStates+parNMPC.nInputs)*parNMPC.N;
parNMPC.nConstr     = parNMPC.nStates*parNMPC.N;

% Initial values
parNMPC.uprev       = [0.0 0.0]';  % Def: u = [u_vtg, u_egr]
[~, linearOp]       = getLinearModel(parNMPC.uprev(1), parNMPC.uprev(2));
parNMPC.x0          = linearOp.xOP;
parNMPC.y0          = linearOp.yOP;
parNMPC.optVars0    = repmat([parNMPC.uprev; parNMPC.x0],parNMPC.N,1);
parNMPC.lambda0     = zeros(parNMPC.nConstr,1);
parNMPC.mu0         = zeros(parNMPC.nOptVars,1);
% Remark: Compared to the lecture, qpOases has a different interface for
% constraints. Equality constraints do not exist, and for inequality
% constraints, there exists an interface for direct optVars constraints
% (lb/ub) and an interface for linear constraints (lbA/ubA) (see
% qpOases Manual). The qpOases QP solution contains one dual-multiplier
% "lambda" vector. Its first nOptVars entries are the dual multipliers for
% the direct optVars constraints (lambda_OptVars), the remaining nConstr
% entries belong to the linear constraints (lambda_AOptVars). For our
% problem, we use the direct optVars constraints for our control input
% inequality constraints and the linear ones for the multiple shooting
% equality constraints (lbA = ubA). Therefore, lambda_AOptVars are the
% dual-multipliers (dual variables) of the QP equality constraints and used
% to update the dual-multipliers "lambda" of the NLP. lambda_OptVars are
% the dual-multipliers of the QP inequality constraints and used to update
% the dual-multipliers "mu" of the NLP.


% Lower and upper bounds on optimization variable vector
% (Bounds are stated for the following optVars definition in NLP:
% optVars = [ u(0|k), x(1|k), u(1|k), x(2|k), ..., u(N-1|k), x(N|k) ])
parNMPC.lbx = repmat([0 0 -1e10 -1e10 -1e10 -1e10 -1e10]',parNMPC.N,1);
parNMPC.ubx = repmat([1 1  1e10  1e10  1e10  1e10  1e10]',parNMPC.N,1);

% The parNMPC struct has to be handed in and will be used for the
% parametrization of your controller. Therefore it will be saved.
save('parNMPC','parNMPC');

% Delay Transfer function
m = 2; % length of model output vector y
Td = [0.1 0.3];
tau = [round(Td(1)/parNMPC.Ts) round(Td(2)/parNMPC.Ts)]; % Delay vector of size m (tau & nu combined in tau vector)
    
    A_Delay = [];
    B_Delay = [];
    C_Delay = [];
    for i = 1:m
        A_Delay     = blkdiag(A_Delay,diag(ones(tau(i)-1,1),1));
        B_Delay_tmp = zeros(tau(i),m);
        B_Delay_tmp(tau(i),i) = 1;
        B_Delay     = [B_Delay;B_Delay_tmp];
        C_Delay_tmp = zeros(m,tau(i));
        C_Delay_tmp(i,1) = 1;
        C_Delay     = [C_Delay,C_Delay_tmp];
    end
    D_Delay = zeros(m,m);
    sysD_Delay = ss(A_Delay,B_Delay,C_Delay,D_Delay,parNMPC.Ts);

%% Simulation Environment Parameters
% The reference is defined as a series of steps defined by a time and a
% reference value vector, which contains the time and reference values of
% each step. These values are used in a Lookup-Table in simulink.
% Therefore, the vectors need to have two or more entries and the time
% vectors have to be monotonically increasing.
% 
% In case of a single step NMPC, the simulation jumps from its
% initialization values defined in the parNMPC struct to the first entry
% of the reference vectors at time t = 0s. A second entry is needed in the
% reference vecotrs in order to be used in a lookup table, however, since
% the simulation time is set to less than one NMPC sampling time, this
% entry is irrelevant.

% Reference Definition
parSim.ref.p_im.time = [0    3    4    7    8    11   12   24.9  25   ...
    28.1 32   40   44   48   49   50];
parSim.ref.p_im.data = [1.05 1.05 1.35 1.35 1.10 1.10 1.05 1.05  1.25 ...
    1.25 1.15 1.15 1.10 1.10 1.05 1.05]*1e5;
parSim.ref.x_bg.time = [0    4    12   16   16.1 20   24   28    32   ...
    36   48   49   50];
parSim.ref.x_bg.data = [0.00 0.00 0.05 0.05 0.10 0.05 0.05 0.25  0.25 ...
    0.15 0.15 0.00 0.00];

% Parameters
parSim.Tsim = parSim.ref.p_im.time(end);


%% Run Simulink NMPC Simulation

% % Compile S-functions, needed for the Simulink simulation
fprintf('\nGenerating S-functions... \n\n');
% The following function creates the mex files needed by the S-functions 
% in the simulink environment and saves them in the folder "sFunctions".
% Furhter, for every mex file, C code is generated. This code will be
% later on used to run the derived simulation on a real-time-processor.
%
% This function works with the MinGW C (not C++) Compiler to generate mex
% files. It can be installed under HOME-->Add-Ons. After installation the
% mex setup has to be configured accordingly. This can be done in the 
% command window with the command "mex -setup".
%createSfunctions(qp_W,qp_gradJ,qp_gradhT,qp_h);
% Add S-functions to working path 
%addpath('sFunctions');
%fprintf('\nDone\n');


% % Run Simulink simulation
% In order to run the simulink simulation, the simulink template "NMPC.slx"
% has to be completed to a running simulation.
fprintf('\nRunning Simulink simulation... ');
simout = sim('NMPC','SaveOutput','on');
fprintf('Done\n');
%%
% % Create Simulation Plot
% extract Simulation Results from Simulink
t_u_Simulink = simout.u.Time;
u1_Simulink  = simout.u.Data(:,1);
u2_Simulink  = simout.u.Data(:,2);
t_y_Simulink = simout.y.Time;
y1_Simulink  = simout.y.Data(:,1);
y2_Simulink  = simout.y.Data(:,2);
t_yROM_Simulink = simout.y_ROM.Time;
yROM1_Simulink  = simout.y_ROM.Data(:,1);
yROM2_Simulink  = simout.y_ROM.Data(:,2);
% Plot: Simulation Results of Matlab and Simulink Simulation
[fig1, ax] = RefTrajFigInit(1,parSim);
figure(fig1)
subplot(ax(1));
plot(t_y_Simulink,y1_Simulink*1e-5,'-');
plot(t_yROM_Simulink,yROM1_Simulink*1e-5,'-');
legend('Reference','Measured','ROM','location','se');
ylabel('p_{im} [bar]');
subplot(ax(2));
plot(t_y_Simulink,y2_Simulink*100,'-');
plot(t_yROM_Simulink,yROM2_Simulink*100,'-');
ylabel('x_{bg} [%]');
subplot(ax(3));
stairs(t_u_Simulink,u1_Simulink*100,'-');
ylabel('u_{vtg} [%]');
subplot(ax(4));
stairs(t_u_Simulink,u2_Simulink*100,'-');
ylabel('u_{egr} [%]');


%% Functions

function [fig,ax] = RefTrajFigInit(num,parSim)
% Figure Initialization with reference values for single step simulations.
%
grey = 0.8*[1 1 1];
fig = figure(num); clf;
ax(1) = subplot(2,2,1); hold on; box on; grid on;
plot(parSim.ref.p_im.time,parSim.ref.p_im.data*1e-5,...
    '--','color',grey,'LineWidth',2);
ylabel('p_{im} [bar]'); ylim([0.95 1.4]);
ax(1).ColorOrderIndex = 1; % restart color order
ax(2) = subplot(2,2,2); hold on; box on; grid on;
plot(parSim.ref.x_bg.time,parSim.ref.x_bg.data*100,...
    '--','color',grey,'LineWidth',2);
ylabel('x_{bg} [-]'); ylim([0 0.25]*100);
ax(2).ColorOrderIndex = 1; % restart color order
ax(3) = subplot(2,2,3); hold on; box on; grid on;
xlabel('time [s]');
ylabel('u_{vtg} [%]'); ylim([-0.1 1.1]*100);
ax(3).ColorOrderIndex = 1; % restart color order
ax(4) = subplot(2,2,4); hold on; box on; grid on;
xlabel('time [s]');
ylabel('u_{egr} [%]'); ylim([-0.1 1.1]*100);
ax(4).ColorOrderIndex = 1; % restart color order
linkaxes(ax,'x');
xlim([0 parSim.Tsim]);
drawnow;
end

% % EOF