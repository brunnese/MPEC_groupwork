%**************************************************************************
%   Model Predictive Engine Control                                       *
%   Spring 2021, IDSC, ETH Zurich                                         *
%   Group Work 01                                                TEMPLATE *
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

%% Task a): Create CasADi functions for NLP with SQP & Multiple Shooting
% Based on the solution of PS8, Ex 1b, generate the needed casADi functions
% using the ROM function. Wrap your solution in the function called here.

[qp_W,qp_gradJ,qp_gradhT,qp_h,FJk] = createCasadiFunctions(parNMPC);

%% NMPC Parameters for the Simulation

% Algorithm parameters
parNMPC.Q1_val      = 1e-6;        % Weighting on p_IM tracking error
parNMPC.Q2_val      = 1e4;         % Weighting on x_bg tracking error
parNMPC.R1_val      = 1e4;         % Weighting on change in u_vtg
parNMPC.R2_val      = 1e4;         % Weighting on change in u_egr
parNMPC.nSQP        = 10;          % Maximum SQP iterations
parNMPC.alphaSQP    = 0.3;         % Step length for damped Newton's method
parNMPC.QPMaxIter   = 1e3;         % Max. number of QP iterations
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
parSim.ref.p_im.time = [0 parNMPC.Ts-1e-3];
parSim.ref.p_im.data = [1.2 1.2]*1e5;
parSim.ref.x_bg.time = [0 parNMPC.Ts-1e-3];
parSim.ref.x_bg.data = [0.1 0.1];

% Parameters
parSim.Tsim = parSim.ref.p_im.time(end);

%% Task b & c): Run Matlab simulation for various nSQP

fprintf('Running Matlab simulations... \n');
% Task 1: Try various numbers of SQP iterations in a Matlab single step
% simulation and compare the results. In order to do so, overwrite the
% number of SQP steps with the pre-defined values saved in the nSQP_vec
% vector in a for-loop. Save the original value for a later comparison of
% matlab and simulink simulation (Task 2).
[fig1,ax,tgrid] = SingleStepFigInit(1,parNMPC,parSim);
nSQP_vec  = [1 2 5 10 1e3];
nSQP_orig = parNMPC.nSQP;
for ii = 1:numel(nSQP_vec)
    
    % Define max. number of SQP steps
    parNMPC.nSQP = nSQP_vec(ii);
    
    % Run single step NMPC in Matlab
    % (This function has to be created first.)
    out = NMPC_Matlab_singlestep(qp_W,qp_gradJ,qp_gradhT,qp_h,FJk,...
                                 parNMPC,parSim);
    
    % Extract solution
    y1Opt = [parNMPC.x0(2); out.optVarsPred(4:7:end)];
    y2Opt = (1-[parNMPC.x0(4); out.optVarsPred(6:7:end)]/0.23142);
    u1Opt = [out.optVarsPred(1:7:end);NaN];
    u2Opt = [out.optVarsPred(2:7:end);NaN];
    
    % Add predicted trajectory to figure
    figure(fig1);
    if ii==numel(nSQP_vec)
        subplot(ax(1)); plot(tgrid,y1Opt*1e-5,'k','LineWidth',1.5);
        subplot(ax(2)); plot(tgrid,y2Opt*100,'k','LineWidth',1.5);
        subplot(ax(3)); stairs(tgrid,u1Opt*100,'k','LineWidth',1.5);
        subplot(ax(4)); stairs(tgrid,u2Opt*100,'k','LineWidth',1.5);    
    else
        subplot(ax(1)); plot(tgrid,y1Opt*1e-5,'LineWidth',1);
        subplot(ax(2)); plot(tgrid,y2Opt*100,'LineWidth',1);
        subplot(ax(3)); stairs(tgrid,u1Opt*100,'LineWidth',1);
        subplot(ax(4)); stairs(tgrid,u2Opt*100,'LineWidth',1);
    end % if
    drawnow;
    
end % for
subplot(ax(1));
legendCell = cellstr(num2str(nSQP_vec', 'N_{SQP} = %-d'));
legend({'ref',legendCell{1:end-1},'N_{SQP} = Inf'},'location','nw');

% In order to complete task c, vary the horizon N (parNMPC.N) and the NMPC
% sampling time (parNMPC.Ts), and rerun the script up to here.

%% Task d & e): Compare Simulink with Matlab Simulation

% % Run Matlab simulation again with original number of SQP steps
% in order to compare it with the Simulink simulation
parNMPC.nSQP = nSQP_orig;
out = NMPC_Matlab_singlestep(qp_W,qp_gradJ,qp_gradhT,qp_h,...
                             FJk,parNMPC,parSim);
y1Opt = [parNMPC.x0(2); out.optVarsPred(4:7:end)];
y2Opt = (1-[parNMPC.x0(4); out.optVarsPred(6:7:end)]/0.23142);
u1Opt = [out.optVarsPred(1:7:end);NaN];
u2Opt = [out.optVarsPred(2:7:end);NaN];
fprintf('\nDone\n');


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
createSfunctions(qp_W,qp_gradJ,qp_gradhT,qp_h);
% Add S-functions to working path 
addpath('sFunctions');
fprintf('\nDone\n');


% % Run Simulink simulation
% In order to run the simulink simulation, the simulink template "NMPC.slx"
% has to be completed to a running simulation.
fprintf('\nRunning Simulink simulation... ');
simout = sim('NMPC','SaveOutput','on');
fprintf('Done\n');


% % Create Comparison Plots
% extract Simulation Results from Simulink
y1Opt_Simulink = [simout.x0.Data(1,2) ...
    simout.optVars.Data(1,4:(parNMPC.nStates+parNMPC.nInputs):end)]';
y2Opt_Simulink = 1-[simout.x0.Data(1,4) simout.optVars.Data(...
    1,6:(parNMPC.nStates+parNMPC.nInputs):end)]'/0.23142;
u1Opt_Simulink = [simout.optVars.Data(...
    1,1:(parNMPC.nStates+parNMPC.nInputs):end) NaN]';
u2Opt_Simulink = [simout.optVars.Data(...
    1,2:(parNMPC.nStates+parNMPC.nInputs):end) NaN]';
% Plot 2: Simulation Results of Matlab and Simulink Simulation
[fig2, ax] = SingleStepFigInit(2,parNMPC,parSim);
figure(fig2)
subplot(ax(1));
plot(tgrid,y1Opt*1e-5);
plot(tgrid,y1Opt_Simulink*1e-5,'--');
legend('Reference','Matlab','Simulink','location','se');
ylabel('p_{im} [bar]');
subplot(ax(2));
plot(tgrid,y2Opt*100);
plot(tgrid,y2Opt_Simulink*100,'--');
ylabel('x_{bg} [%]');
subplot(ax(3));
stairs(tgrid,u1Opt*100);
stairs(tgrid,u1Opt_Simulink*100,'--');
ylabel('u_{vtg} [%]');
subplot(ax(4));
stairs(tgrid,u2Opt*100);
stairs(tgrid,u2Opt_Simulink*100,'--');
ylabel('u_{egr} [%]');
% Plot 3: Difference of Matlab and Siulink Simulation
figure(3); clf;
subplot(2,2,1); box on;
plot(tgrid,(y1Opt-y1Opt_Simulink)*1e-5);
ylabel('p_{im,mat} - p_{im,simu} [bar]');
grid on;
subplot(2,2,2); box on;
plot(tgrid,(y2Opt-y2Opt_Simulink)*100);
ylabel('x_{bg,mat} - x_{bg,simu} [%]');
grid on;
subplot(2,2,3); box on;
stairs(tgrid,(u1Opt-u1Opt_Simulink)*100);
xlabel('time [s]');
ylabel('u_{vtg,mat} - u_{vtg,simu} [%]');
grid on;
subplot(2,2,4); box on;
stairs(tgrid,(u2Opt-u2Opt_Simulink)*100);
xlabel('time [s]');
ylabel('u_{egr,mat} - u_{egr,simu} [%]');
grid on;


%% Functions

function [fig,ax,tgrid] = SingleStepFigInit(num,parNMPC,parSim)
% Figure Initialization with reference values for single step simulations.
%
grey = 0.8*[1 1 1];
tgrid = linspace(0,parNMPC.Ts*parNMPC.N,parNMPC.N+1);

fig = figure(num); clf;
ax(1) = subplot(2,2,1); hold on; box on; grid on;
plot(tgrid,parSim.ref.p_im.data(1)*ones(size(tgrid))*1e-5,...
    '--','color',grey,'LineWidth',2);
ylabel('p_{im} [bar]'); ylim([0.95 1.4]);
ax(1).ColorOrderIndex = 1; % restart color order
ax(2) = subplot(2,2,2); hold on; box on; grid on;
plot(tgrid,parSim.ref.x_bg.data(1)*ones(size(tgrid))*100,...
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
xlim([0 parNMPC.Ts*parNMPC.N]);
drawnow;
end

% % EOF