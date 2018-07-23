include("DPM_BIRL.jl")
using POMDPModels
using POMDPToolbox

srand(1)
n_agents = 2
traj_per_agent = 30
iterations = 200
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

# χ = χ[1:300]
# policies[1], policies[2] = policies[2], policies[1]
# mdps[1], mdps[2] = mdps[2], mdps[1]
# deleteat!(policies,1)
# deleteat!(mdps,1)
# srand(1)
# mdp, policy = DPMBIRL.generate_gridworld(10,10,γ=0.9)
# χ = DPMBIRL.generate_trajectories(mdp, policy, 50)
# ϕ = eye(100)
# learning_rate = 0.1
# confidence = 1.0
#
# policies = [policy]
# mdps = [mdp]
# iterations = 30

raw_mdp = copy(mdps[1])
# raw_mdp.reward_states = Array{GridWorldState}(0)
raw_mdp.reward_values = Array{Float64}(0)

# raw_mdp = mdp


vs = []
for i in 1:size(mdps,1)
	push!(vs, DPMBIRL.policy_evaluation(mdps[i], policies[i], η=.9, π_type=:ϵgreedy))
end
ground_truth = Dict(:policy=>policies, :rewards=>map(x->x.reward_values, mdps), :vs=>vs)


c, _log = DPMBIRL.DPM_BIRL(raw_mdp, ϕ, χ, iterations; α=learning_rate, β=confidence, κ=0.1, ground_truth = ground_truth, verbose = true, update = :langevin_rand, burn_in=30)

for (j, policy) in enumerate(policies)
	v = DPMBIRL.policy_evaluation(mdps[j], policy, η=.9, π_type=:ϵgreedy)
	for (i,r) in enumerate(c.rewards)
		πᵣ = DPMBIRL.solve_mdp(mdps[j], r)
		vᵣ = DPMBIRL.policy_evaluation(mdps[j], πᵣ, η=.9, π_type=:ϵgreedy)
		println("Final EVD for reward $i and policy $j: $(norm(v-vᵣ))")
	end
end

include("DPM_BIRL.jl")
srand(1)
mdp, policy = DPMBIRL.generate_gridworld(10,10,γ=0.9)
v = policy.util[1:end-1]
# v = reshape(reshape(v,(10,10))', (100,1))
vᵣ = DPMBIRL.policy_evaluation(mdp, policy)
norm(v-vᵣ)


using Plots
fig = @gif for ass in log[:assignements]
	histogram(ass)
end
show(fig)

clusters_hist = log[:clusters]
curr_cluster = clusters_hist[190]
curr_reward = curr_cluster.rewards[1]
@enter DPMBIRL.proposal_distribution(curr_reward, DPMBIRL.sample(DPMBIRL.RewardFunction,100), curr_reward.∇𝓛, 0.1)

@gif for evd in log[:EVDs]
	heatmap(evd)
end
