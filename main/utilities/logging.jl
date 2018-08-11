"""
    Simply creates an empty JLD logging file
"""
function prepare_JLD_log_file(path_to_folder, parameters)
    if path_to_folder !== nothing
        update = parameters["Update"]
        seed = parameters["Problem seed"]
        use_clusters = parameters["Use clusters"]
        n_agents = parameters["Number of agents"]
        n_trajectories = parameters["Number of trajectories per agent"]


        path_to_file = "$path_to_folder/$(update)_$(seed)_$(use_clusters)_$(n_agents)_$(n_trajectories).jld"
        f = jldopen(path_to_file, "w")
        close(f)
        return path_to_file
    else
        return nothing
    end
end
