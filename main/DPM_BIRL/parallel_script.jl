addprocs(Sys.CPU_CORES-1-nprocs())
@everywhere include("/home/edoardo/Documents/Masters Work/Dissertation/code/main/DPM_BIRL/DPM_BIRL.jl")
@everywhere using POMDPToolbox
@everywhere using JLD


@everywhere function gather_results(seed, n_agents, traj_per_agent, iterations)

	problem_seed = 1
	srand(problem_seed)

	iterations = iterations
	confidence = 1.0
	burn_in = 200
	ϕ = eye(100)
	χ = Array{MDPHistory}(0)
	mdps = []
	policies = []
	for i in 1:n_agents
		mdp, policy = DPMBIRL.generate_gridworld(10,10,γ=0.9)
		χₐ = DPMBIRL.generate_trajectories(mdp, policy, traj_per_agent)
		push!(χ, χₐ...)
		push!(mdps, mdp)
		push!(policies, policy)
	end

	use_clusters = true
	concentration = 1.0

	raw_mdp = copy(mdps[1])
	raw_mdp.reward_values = Array{Float64}(0)

	ground_truth = Dict(:policy=>policies, :rewards=>map(x->x.reward_values, mdps), :vs=>vs)

	parameters = Dict("Number of agents"=>n_agents, "Number of trajectories per agent"=>traj_per_agent,
												"Confidence"=>confidence, "Concentration"=>concentration,
												"Problem specs"=>"10x10, features = states", "Update"=>"Langevin",
												"Burn in"=>burn_in, "Use clusters"=>use_clusters, "Problem seed"=>problem_seed)

	# folder = prepare_log_folder(pwd()*"/results", parameters)

	logs = []
	# prepare_log_file(folder, seed)
	τ = DPMBIRL.LoggedFloat(.8)
	c, _log = DPMBIRL.DPM_BIRL(raw_mdp, ϕ, χ, iterations; τ=τ, β=confidence, κ=concentration,
								ground_truth = ground_truth, verbose = true, update = :langevin_rand,
								burn_in=burn_in, use_clusters=use_clusters, seed=seed, path_to_folder="$(pwd())/results", parameters=parameters)
	parameters["seed"] = seed
	_log
end

# Changing number trajectories per agent: [10, 20, 40, 80]
# Changing number of agents: [1,2,4,6,8]
# changing confidence: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
seeds = [1,2,3,4,5]
iterations = 2000
for n_agents in [2,3,4]
	for traj_per_agent in [10,20,40]
		futures = Dict()
		# Launch all processes
		for (index, seed) in enumerate(seed)
			futures[seed] = @spawn gather_results(seed, n_agents, traj_per_agent, iterations)
		end
		# wait for processes
		for key in keys(futures)
			futures[key] = fetch(futures[key])
		end
		for seed in seeds
			println("File saved at: $(pwd())/results/$(seed)_$(n_agents)_$(traj_per_agent).jld")
			save("$(pwd())/results/$(seed)_$(n_agents)_$(traj_per_agent).jld", "results", futures[seed])
			println("Saved results for $(n_agents) agents and $(traj_per_agent) trajectories per agent")
		end
	end
end


@everywhere g(x) = x^2

futures = Dict()
for (i,x) in enumerate([1,2,3,4,5])
	futures[i] = @spawn g(x)
end

for key in keys(futures)
	futures[key] = fetch(futures[key])
end
futures
