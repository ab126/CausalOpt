function run_liegeois_batch(input_path, output_path, repo_path)
% File-based batch adapter around the existing project-owned Engine bridge.
input = load(input_path);
[x_sol, infos, used_options, C] = run_liegeois_engine( ...
    input.X, input.p, repo_path, input.lambda_value, input.gamma, ...
    input.maxiter, input.rho_max, input.abstol, input.reltol, ...
    input.compute_relative_duality_gap, input.compute_primal_variables, input.verb);
matlab_version = version;
save(output_path, 'x_sol', 'infos', 'used_options', 'C', 'matlab_version', '-v7');
end
