function [x_sol, infos, used_options, C] = run_liegeois_engine(X, p, repo_path, lambda, gamma, maxiter, rho_max, abstol, reltol, compute_relative_duality_gap, compute_primal_variables, verb)
% Thin project-owned bridge to the unchanged upstream reference algorithm.
addpath(repo_path);
X = double(X);                 % T-by-n, matching xcorr(X',...) in upstream Test.m
[Nt, n] = size(X); %#ok<ASGLU>
temp = xcorr(X, p, 'biased');
C_line = zeros(n,n,2*p+1);
for lag = -p:p
    C_line(:,:,lag+p+1) = reshape(temp(lag+p+1,:),n,n);
end
C = zeros(n*(p+1));
for i = 1:p+1
    for j = 1:p+1
        C(((i-1)*n+1):i*n,((j-1)*n+1):j*n) = C_line(:,:,(j-i)+p+1);
    end
end
problem.n = n; problem.p = p; problem.C = C;
options.lambda = lambda; options.gamma = gamma; options.maxiter = maxiter;
options.rho_max = rho_max; options.abstol = abstol; options.reltol = reltol;
options.compute_relative_duality_gap = logical(compute_relative_duality_gap);
options.compute_primal_variables = logical(compute_primal_variables);
options.verb = logical(verb);
[x_sol, infos, used_options] = ADMM_sparse_lowrank_AR(problem, [], options);

% MATLAB Engine for Python cannot transfer MATLAB sparse arrays.  Densify
% only the transfer representation after the upstream estimator has
% completed; this does not alter estimator values or support semantics.
if isfield(x_sol, 'L') && issparse(x_sol.L)
    x_sol.L = full(x_sol.L);
end
if isfield(x_sol, 'S') && issparse(x_sol.S)
    x_sol.S = full(x_sol.S);
end
if isfield(x_sol, 'Omega')
    omega_support_before = full(x_sol.Omega ~= 0);
    if issparse(x_sol.Omega)
        x_sol.Omega = full(x_sol.Omega);
    end
    assert(isequal(omega_support_before, x_sol.Omega ~= 0), ...
        'Liéois bridge changed the Omega nonzero pattern during conversion.');
end
if isfield(x_sol, 'Delta') && issparse(x_sol.Delta)
    x_sol.Delta = full(x_sol.Delta);
end
if issparse(C)
    C = full(C);
end

% Guard against sparse fields added to the upstream output structs without
% changing the upstream repository or optimization code.
x_sol = full_sparse_struct_fields(x_sol);
infos = full_sparse_struct_fields(infos);
used_options = full_sparse_struct_fields(used_options);
% Release the upstream MEX before returning so Windows does not leave its
% tracked binary mapped while the integration test checks repository state.
clear projectSortC
end

function value = full_sparse_struct_fields(value)
% Recursively densify sparse numeric/logical fields in project-bound structs.
if issparse(value) && (isnumeric(value) || islogical(value))
    value = full(value);
elseif isstruct(value)
    names = fieldnames(value);
    for element = 1:numel(value)
        for field = 1:numel(names)
            name = names{field};
            value(element).(name) = full_sparse_struct_fields(value(element).(name));
        end
    end
end
end
