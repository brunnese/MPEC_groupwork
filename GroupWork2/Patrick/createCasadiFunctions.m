function [qp_W,qp_gradJ,qp_gradhT,qp_h,FJk,lb,ub] = createCasadiFunctions(options)
import casadi.*
%% Definition of model dynamics and objective function (continuous)
% States and inputs
x       = MX.sym('x',5,1);
u       = MX.sym('u',2,1);
nStates = length(x);
nInputs = length(u);

% System dynamics
[xdot, y] = ROModel(x, u)
nOutputs = length(y);
% Objective function with needed Parameters
Q1  = MX.sym('Q1',1,1);
Q2  = MX.sym('Q2',1,1);
R1  = MX.sym('R1',1,1);
R2  = MX.sym('R2',1,1);
du  = MX.sym('du',2,1);
r   = MX.sym('r',2,1);

J       = Q1*(y(1)-r(1))^2 + Q2*(y(2)-r(2))^2 + R1*du(1)^2 + R2*du(2)^2;
% discrete casadi function
% unterschied zu ps08b hier ist die kostenfunktion bereits diskretisiert
fJDisc  = Function('fJDisc', {x,du,r,[Q1;Q2;R1;R2]}, {J});

% Create CasADI functions sys dynamics
fxdot 	= Function('fxdot', {x,u}, {xdot});


%% Integration/discretization using RK4
xStart  = MX.sym('xStart',nStates,1);   % Initial condition of integration
u       = MX.sym('u',nInputs,1);        % Control input
TRK4    = options.Ts/options.nRK4;      % Step size of each RK4 interval

% Loop over intervals
xEnd    = xStart;                       % Initialization
JEnd    = 0;                            % Initialization
% diskretisierung der traj
for l = 1:options.nRK4
    k1x = fxdot(xEnd, u);
    k2x = fxdot(xEnd + TRK4/2 * k1x, u);
    k3x = fxdot(xEnd + TRK4/2 * k2x, u);
    k4x = fxdot(xEnd + TRK4 * k3x, u);
    xEnd = xEnd + TRK4/6 * (k1x + 2*k2x + 2*k3x + k4x);
    
end % for

% Create CasADi functions
fxDisc  = Function('fxDisc', {xStart,u}, {xEnd});

%% Construct NLP
% Initialization
optVars     = [];   % Vector of optimization variables (i.e. states and inputs)
optVars0    = [];   % Initial guess for optimization variables
lb          = [];   % Lower bound of optimization variables
ub          = [];   % Upper bound of optimization variables
Jk          =  0;   % Initialization of objective function
g           = [];   % (In-)equality constraints
lbg         = [];   % Lower bound on (in-)equality constraints
ubg         = [];   % Upper bound on (in-)equality constraints

% Pre-define CasADi variables
U = MX.sym('U_',nInputs,1,options.N);
S = MX.sym('S_',nStates,1,options.N+1);
p = MX.sym('p',nStates+nInputs+nOutputs+4) % xo,uprev,r,Q1,Q2;R1;R2 / 13 variablen
% Construct NLP step-by-step
for k = 1:options.N+1
    
    % System dynamics and objective function
    if k==options.N+1
        % Skip, because at final time step, no further integration of
        % system dynamics necessary.
    else
        % Integration of system dynamics and objective function
        if k==1
            % Hardcoded initial condition
            XEnd = fxDisc(p{1:5},U{k});
            Jk = fJDisc(p{1:5},U{k}-p{6:7},p{8:9},p{10:13});
        else
            XEnd = fxDisc(S{k},U{k});
            Jk = Jk + fJDisc(S{k},U{k}-U{k-1},p{8:9},p{10:13});
        end % if
        
        % Add equality constraint for continuity (i.e. closing gaps) das haben wir genau so aber 5 states:
        g   = [g; XEnd-S{k+1}];
        lbg = [lbg; 0; 0;0;0;0];
        ubg = [ubg; 0; 0;0;0;0];
        
    end % if
    
    % States
    if k==1
        % Skip, because we have hardcoded the initial condition above
    else
        % Add states to vector of optimization variables
        optVars = [optVars; S{k}];
        
        % Lower- and upper bounds for states haben wir noch keine
        lb = [lb; -Inf; -Inf; -Inf; -Inf; -Inf];
        ub = [ub; Inf; Inf; Inf; Inf; Inf];
        
        % Add initial guess of states nehme wieder den vorherig berechneten
        optVars0 = [optVars0; p{6:7}];
        
    end % if
    
    % Inputs
    if k==options.N+1
        % Skip, because no control input at final time step
    else
        % Add inputs to vector of optimization variables
        optVars = [optVars; U{k}];
        
        % Lower- and upper bounds for inputs
        lb = [lb; 0; 0];
        ub = [ub; 1; 1];
        
        % Add initial guess for inputs
        optVars0 = [optVars0; 0];
        
    end % if
    
end % for
%% Preparation for SQP

% Construct Lagrange function
lambda  = MX.sym('lambda',numel(g));
L       = Jk + lambda'*g;

% Calculate the hessian and jacobian of the objective function and the 
% (in-)equality constraints using CasADi
W       = hessian(Jk,optVars);
%W       = hessian(L,optVars);
gradJ   = jacobian(Jk,optVars)';

gradhT 	= jacobian(g,optVars);

% Create CasADi functions
qp_W        = Function('qp_W',{optVars,[p;lambda]},{W});
qp_gradJ  	= Function('qp_gradJ',{optVars,p},{gradJ});
qp_gradhT	= Function('qp_gradhT',{optVars,p},{gradhT});
qp_h     	= Function('qp_h',{optVars,p},{g});
FJk         = Function('FJk',{optVars,p},{Jk});

end % function

% EOF