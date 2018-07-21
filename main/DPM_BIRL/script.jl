include("DPM_BIRL.jl")
using POMDPToolbox

srand(1)
n_agents = 1
traj_per_agent = 50
iterations = 100
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
ϕ = eye(100)
learning_rate = 0.1
confidence = 0.8

c, EVD, log = DPMBIRL.DPM_BIRL(mdps[1], ϕ, χ, iterations; α=learning_rate, β=confidence, κ=0.1, ground_policy = policies[1], verbose = true, update = :ML)

for (i,r) in enumerate(c.rewards)
	# Need to change this to account for features
	for (j, policy) in enumerate(policies)
		v = DPMBIRL.policy_evaluation(mdps[j], policy)
		πᵣ = DPMBIRL.solve_mdp(mdps[j], r)
		vᵣ = DPMBIRL.policy_evaluation(mdps[j], πᵣ)
		push!(EVD, norm(v-vᵣ))
		println("Final EVD for reward $i and policy $j: $(EVD[end])")
	end
end


using Plots
fig = @gif for ass in log[:assignements]
	histogram(ass)
end
show(fig)


histogram(log[:assignements][1], nbins=5)

rewards = rewards_matrix(mdp)
heatmap(rewards')
heatmap(reshape(θ.values, (10,10)))

fig
