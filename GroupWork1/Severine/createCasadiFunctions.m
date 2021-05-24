function [qp_W,qp_gradJ,qp_gradhT,qp_h,lb,ub] = createCasadiFunctions(parNMPC)
import casadi.*
%% Definition of model dynamics and objective function (continuous)

% States and inputs
nStates  = parNMPC.nStates;
nInputs  = parNMPC.nInputs;
nOutputs = parNMPC.nOutputs;

x        = MX.sym('x',nStates,1);
r        = MX.sym('r',2,1);
du       = MX.sym('du',nInputs,1);
u       = MX.sym('u',nInputs,1);        % Control input
Q1       = MX.sym('Q1');
Q2       = MX.sym('Q2');
R1       = MX.sym('R1');
R2       = MX.sym('R2');

% Pre-define parameters
p = MX.sym('p',nStates+nInputs+nOutputs+4); %p = [x0;uprev;r;Q1;Q2;R1;R2]

[xdot, y] = ROModel(x, u);

% System dynamics
x1dot   = xdot(1); % n_tc
x2dot   = xdot(2); % p_im
x3dot   = xdot(3); % p_em
x4dot   = xdot(4); % F_im
x5dot   = xdot(5); % F_em
xdot    = [x1dot; x2dot; x3dot; x4dot; x5dot];

% Objective function
J       = Q1*(y(1) - r(1))^2 + Q2*(y(2) - r(2))^2 ...
    + R1*(du(1))^2 + R2*(du(2))^2;

% Create CasADI functions
fxdot 	= Function('fxdot', {x,u}, {xdot});
fJ      = Function('fJ', {x,du,r,[Q1;Q2;R1;R2]}, {J});

%% Integration/discretization using RK4
xStart  = MX.sym('xStart',nStates,1);   % Initial condition of integration

TRK4    = parNMPC.Ts/parNMPC.nRK4;      % Step size of each RK4 interval

% Loop over intervals
xEnd    = xStart;                       % Initialization
                        
for l = 1:parNMPC.nRK4
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
optVars  	= [];   % Vector of optimization variables (i.e. states and inputs)
optVars0  	= [];   % Initial guess for optimization variables
lb          = [];   % Lower bound of optimization variables
ub          = [];   % Upper bound of optimization variables
Jk          =  0;   % Initialization of objective function
h           = [];   % Equality constraints

% Pre-define CasADi variables
U = MX.sym('U_',nInputs,1,parNMPC.N);
S = MX.sym('S_',nStates,1,parNMPC.N+1);

% Construct NLP step-by-step
for k = 1:parNMPC.N+1
    
    % System dynamics and objective function
    if k==parNMPC.N+1
        % Skip, because at final time step, no further integration of 
        % system dynamics necessary.
    else
        % Integration of system dynamics and objective function
        if k==1
            % Hardcoded initial condition
            XEnd = fxDisc(p(1:5),U{k});
            Jk = Jk + fJ(p(1:5),U{k}-p(6:7),p(8:9),p(10:13));
        else
            XEnd = fxDisc(S{k},U{k});
            Jk = Jk + fJ(S{k},U{k}-U{k-1},p(8:9),p(10:13));
        end % if
        
        % Add equality constraint for continuity (i.e. closing gaps):
        h = [h; XEnd-S{k+1}];
        
    end % if
    
    % States
    if k==1
        %skip because initial conditions hard coded
    else
        % Add states to vector of optimization variables
        optVars = [optVars; S{k}];   
    end % if
    
    % Inputs
    if k==parNMPC.N+1
        % Skip, because no control input at final time step
    else
       % Add inputs to vector of optimization variables
       optVars = [optVars; U{k}];    
    end % if  
end % for

%% Preparation for SQP

% Construct Lagrange function
lambda  = MX.sym('lambda',numel(h));
L       = Jk + lambda'*h;

% Calculate the hessian and jacobian of the objective function and the 
% (in-)equality constraints using CasADi
W       = hessian(L,optVars);
gradJ   = jacobian(Jk,optVars)';
gradhT 	= jacobian(h,optVars);

% Create CasADi functions
qp_W        = Function('qp_W',{optVars,[p;lambda]},{W});
qp_gradJ  	= Function('qp_gradJ',{optVars,p},{gradJ});
qp_gradhT	= Function('qp_gradhT',{optVars,p},{gradhT});
qp_h     	= Function('qp_h',{optVars,p},{h});
%FJk         = Function('FJk', {optVars,p},{fJ};

end % function

% EOF
