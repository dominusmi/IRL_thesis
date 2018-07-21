addprocs(Sys.CPU_CORES-1-nprocs())
@everywhere include("/home/edoardo/Documents/Masters Work/Dissertation/code/main/DPM_BIRL/DPM_BIRL.jl")
@everywhere using POMDPToolbox
@everywhere using JLD


@everywhere function gather_results(n_agents, traj_per_agent, confidence, iterations)
	srand(n_agents*traj_per_agent^2)
	χ = Array{MDPHistory}(0)
	mdps = []
	policies = []
	ϕ = eye(100)
	learning_rate = 0.1

	for i in 1:n_agents
		mdp, policy = DPMBIRL.generate_gridworld(10,10,γ=0.9)
		χₐ = DPMBIRL.generate_trajectories(mdp, policy, traj_per_agent)
		push!(χ, χₐ...)
		push!(mdps, mdp)
		push!(policies, policy)
	end

	c, EVD, log = DPMBIRL.DPM_BIRL(mdps[1], ϕ, χ, iterations; α=learning_rate, β=confidence, κ=0.1, ground_policy = policies[1], verbose = true, update = :ML)

	EVD_matrix = zeros( size(c.rewards,1), size(policies,1) )
	for (i,r) in enumerate(c.rewards)
		# Need to change this to account for features
		for (j, policy) in enumerate(policies)
			v = DPMBIRL.policy_evaluation(mdps[j], policy)
			πᵣ = DPMBIRL.solve_mdp(mdps[j], r)
			vᵣ = DPMBIRL.policy_evaluation(mdps[j], πᵣ)
			println("Final EVD for reward $i and policy $j: $(EVD[end])")
			EVD_matrix[i,j] = norm(v-vᵣ)
		end
	end

	log, EVD_matrix
end

# Changing number trajectories per agent: [10, 20, 40, 80]
# Changing number of agents: [1,2,4,6,8]
# changing confidence: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

iterations = 5
for n_agents in [1]
	for traj_per_agent in [10]
		confidences = [.5, .6, .7, .8, .9, 1.0]
		futures = Dict()
		# Launch all processes
		for (index, confidence) in enumerate(confidences)
			futures[confidence] = @spawn gather_results(n_agents, traj_per_agent, confidence, iterations)
		end
		# wait for processes
		for key in keys(futures)
			futures[key] = fetch(futures[key])
		end
		futures["seed"] = n_agents*traj_per_agent^2
		println("File saved at: $(pwd())/results/$(n_agents)_$(traj_per_agent).jld")
		save("$(pwd())/results/$(n_agents)_$(traj_per_agent).jld", "results", futures)
		println("Saved results for $(n_agents) agents and $(traj_per_agent) trajectories per agent")
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
