include("DPM_BIRL.jl")
using POMDPModels
using POMDPToolbox

srand(1)
n_agents = 2
traj_per_agent = 50
iterations = 50
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
							ground_truth = ground_truth, verbose = true, update = :langevin, burn_in=1)


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
Plots.plot!(lhs, label="Likelihood cluster#1", linestyle=:dot, color="blue")
fig = Plots.plot!([ _log[:EVDs][i][2,2] for i in 1:size(_log[:EVDs],1)], label="EVD cluster#2",  linewidth=1.5,color="red")
lhs = [ _log[:likelihoods][i][2] for i in 1:size(_log[:EVDs],1)]
Plots.plot!(lhs, label="Likelihood cluster#2", linestyle=:dot, color="red")
savefig(fig, "EVD-LH_LANG_2_50_wprior_normed.png")


lhs

clusters_hist = log[:clusters]
curr_cluster = clusters_hist[190]
curr_reward = curr_cluster.rewards[1]
@enter DPMBIRL.proposal_distribution(curr_reward, DPMBIRL.sample(DPMBIRL.RewardFunction,100), curr_reward.∇𝓛, 0.1)

@gif for evd in log[:EVDs]
	heatmap(evd)
end



mdp = mdps[2]
policy = policies[2]
states = ordered_states(mdp)
πᵦ = zeros(size(states,1)-1, size(actions(mdp),1))

for s in states[1:end-1]
	si = state_index(mdp,s)
	softmax_denom = sum(exp.(confidence*policy.qmat[si,:]))
	for a in actions(mdp)
		ai = action_index(mdp,a)
		πᵦ[si,ai] = exp(confidence*policy.qmat[si,ai]) / softmax_denom
	end
end

llh = 0.
for trajectory in χ
	# Calculate likelihood trajectory
	log_likelihood = 0.
	traj_size = size(trajectory.state_hist,1)-1
	for (h,state) in enumerate(trajectory.state_hist[1:end-1])
		sₕ = state_index(mdp, state)
		aₕ = action_index(mdp, trajectory.action_hist[h])
		log_likelihood += Base.log(πᵦ[sₕ,aₕ])
	end
	llh += log_likelihood /50
end
llh
