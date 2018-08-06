"""
    Simply creates an empty JLD logging file
"""
function prepare_JLD_log_file(path_to_file, update, seed, use_clusters)
    if path_to_file !== nothing
        path_to_file = "$path_to_file/$(update)_$(seed)_$(use_clusters).jld"
        f = jldopen(path_to_file, "w")
        close(f)
    end
end
