include("DPM_BIRL.jl")
using POMDPModels
using POMDPToolbox

srand(1)
n_agents = 1
traj_per_agent = 50
iterations = 10000
learning_rate = .1
confidence = 1.0
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


raw_mdp = copy(mdps[1])
raw_mdp.reward_values = Array{Float64}(0)

vs = []
for i in 1:size(mdps,1)
	push!(vs, DPMBIRL.policy_evaluation(mdps[i], policies[i], η=.9, π_type=:ϵgreedy))
end
ground_truth = Dict(:policy=>policies, :rewards=>map(x->x.reward_values, mdps), :vs=>vs)



c, _log = DPMBIRL.DPM_BIRL(raw_mdp, ϕ, χ, iterations; α=learning_rate, β=confidence, κ=0.1,
							ground_truth = ground_truth, verbose = false, update = :MH,
							burn_in=100, use_clusters=false, path_to_file="$(pwd())", seed=1)


d = load("MH_false.jld")

keys_d = collect(keys(d))
l_keys = ismatch.(r"likelihood\_[0-9]+", keys_d)
likelihoods = []
for index in 1:sum(l_keys)
	push!(likelihoods, d["likelihood_$index"])
end

Plots.plot(likelihoods)
