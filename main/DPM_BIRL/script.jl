include("DPM_BIRL.jl")
include("../result_explorer.jl")
using POMDPModels
using POMDPToolbox

# ϕ = zeros(100,10)
# for j in 0:9
# 	for i in j*10:(j+1)*10-1
# 		ϕ[i+1,j+1] = 1
# 	end
# end

problem_seed = 1
srand(problem_seed)

n_agents = 1
traj_per_agent = 20
iterations = 300
confidence = 1.0
burn_in = 50
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

use_clusters = false
concentration = 1.0

raw_mdp = copy(mdps[1])
raw_mdp.reward_values = Array{Float64}(0)

vs = []
for i in 1:size(mdps,1)
	push!(vs, DPMBIRL.policy_evaluation(mdps[i], policies[i], η=.9, π_type=:ϵgreedy))
end
ground_truth = Dict(:policy=>policies, :rewards=>map(x->x.reward_values, mdps), :vs=>vs)

parameters = Dict("Number of agents"=>n_agents, "Number of trajectories per agent"=>traj_per_agent,
											"Confidence"=>confidence, "Concentration"=>concentration,
											"Problem specs"=>"10x10, features = states", "Update"=>"Langevin",
											"Burn in"=>burn_in, "Use clusters"=>use_clusters, "Problem seed"=>problem_seed)

# folder = prepare_log_folder(pwd()*"/results", parameters)

logs = []
for seed in 1:1
	# prepare_log_file(folder, seed)
	τ = DPMBIRL.LoggedFloat(.8)
	c, _log = DPMBIRL.DPM_BIRL(raw_mdp, ϕ, χ, iterations; τ=τ, β=confidence, κ=concentration,
								ground_truth = ground_truth, verbose = true, update = :langevin_rand,
								burn_in=burn_in, use_clusters=use_clusters, seed=seed)
	parameters["seed"] = seed
	# summary_statistics(_log, parameters, folder)
	push!(logs, _log)
end

summary, fig = summary_statistics(logs[1][:rewards], parameters)
fig
real = heatmap(reshape(mdps[1].reward_values, (10,10)))
heatmap(reshape(mdps[2].reward_values, (10,10)))
found = heatmap(reshape(summary[:reward_means], (10,10)))

fig = Plots.plot(real, found, layout=(1,2), size=(1600,600), dpi=300, title="")
savefig(fig, "Real vs Mean reward")

for (j, policy) in enumerate(policies)
	v = DPMBIRL.policy_evaluation(mdps[j], policy, η=.9, π_type=:ϵgreedy)
	for (i,r) in enumerate(c.rewards)
		πᵣ = DPMBIRL.solve_mdp(mdps[j], r)
		vᵣ = DPMBIRL.policy_evaluation(mdps[j], πᵣ, η=.9, π_type=:ϵgreedy)
		println("Final EVD for reward $i and policy $j: $(norm(v-vᵣ))")
	end
end

using Plots

fig = Plots.plot([ _log[:EVDs][i][1] for i in 1:size(_log[:EVDs],1)], title="EVD for langevin, 2 agents, 50 trajectories", label="EVD cluster#1", linewidth=1.5, color="blue")
lhs = [ _log[:likelihoods][i][1] for i in 1:size(_log[:EVDs],1)]
Plots.plot(exp.(lhs), label="Likelihood cluster#1", linestyle=:dot, color="blue")
fig = Plots.plot!([ _log[:EVDs][i][2,2] for i in 1:size(_log[:EVDs],1)], label="EVD cluster#2",  linewidth=1.5,color="red")
lhs = [ _log[:likelihoods][i][2] for i in 1:size(_log[:EVDs],1)]
Plots.plot!(lhs, label="Likelihood cluster#2", linestyle=:dot, color="red")
savefig(fig, "EVD-LH_LANG_2_50_wprior_normed.png")


lhs

clusters_hist = log[:clusters]
curr_cluster = clusters_hist[190]
curr_reward = curr_cluster.rewards[1]
@enter DPMBIRL.proposal_distribution(curr_reward, DPMBIRL.sample(DPMBIRL.RewardFunction,100), curr_reward.∇𝓛, 0.1)
