include("DPM_BIRL.jl")
using POMDPModels
using POMDPToolbox

problem_seed = 1
srand(problem_seed)

n_agents = 1
traj_per_agent = 20
iterations = 150
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
concentration = 0.1

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

folder = prepare_log_folder(pwd()*"/results", parameters)

logs = []
for seed in 5:5
	τ = DPMBIRL.LoggedFloat(.8)
	c, _log = DPMBIRL.DPM_BIRL(raw_mdp, ϕ, χ, iterations; τ=τ, β=confidence, κ=concentration,
								ground_truth = ground_truth, verbose = true, update = :langevin_rand,
								burn_in=burn_in, use_clusters=use_clusters, seed=seed)
	parameters["seed"] = seed
	summary_statistics(_log, parameters, folder)
	push!(logs, _log)
end


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

@gif for evd in log[:EVDs]
	heatmap(evd)
end


probs = _log[:acc_prob][:]
probs = probs[probs .< 5]
mean(probs)
Plots.plot(probs, ylim=(0,2), title="Acceptance probs", xlab="Iterations", ylab="Probability")
savefig("acceptance - prior, norm reward")

fig = Plots.plot();
for i in 1:3
	fig = Plots.plot!(map(x->x[1].values[i], _log[:rewards]))
end
fig



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


#### Acceptance Probability ####
temp = abs.(_log[:acc_prob]) .< 3.
acc_prob = _log[:acc_prob][temp]
# temp = abs.(_log[:acc_prob]) .< 1e-3
Plots.plot(acc_prob, ylim=(-.1,2.),
			title="Acceptance probabilities using Langevin update\nNo prior, normalised reward",
			xlabel="Iterations",
			ylabel="P")
savefig("langevin acc prob - prior, norm-reward.png")

#### Likelihoods ####
Plots.plot([ _log[:likelihoods][i][1] for i in 1:size(_log[:likelihoods],1)],
			title="Likelihoods using Langevin update",
			xlabel="Iteration",
			ylabel="Likelihood")
savefig("langevin likelihood - prior, norm reward.png")

#### Change rate ####
Plots.plot([ _log[:changed][i][1] for i in 1:size(_log[:changed],1)],
			title="Accepted Iterations",
			xlabel="Iteration",
			ylabel="Changed")

#### Rewards ####
function log2reward(reward_log)
	rewards = zeros(size(reward_log,1), 100)
	for i in 1:size(reward_log,1)-1
		rewards[i,:] = reward_log[i][1].values
	end
	rewards
end
function plot_reward(reward_log; path_to_save=nothing)
	fig = Plots.plot( log2reward(reward_log), legend=false, ylim=(-0.05, 0.1) )
	Plots.plot!( [burn_in, burn_in], [-1., 2.], color="red", linestyle=:dot, linewidth=2)
	if path_to_save !== nothing
		savefig(path_to_save)
	end
	fig
end

function plot_rewards(reward_logs; plot_args=[], path_to_save=nothing)
	n_rewards = size(reward_logs,1)
	figs = [plot_reward(reward_logs[i]) for i in 1:n_rewards]
	fig = Plots.plot( figs..., legend=false, ylim=(-0.05, 0.1), layout = (n_rewards,1))
	if path_to_save !== nothing
		savefig(path_to_save)
	end
	fig
end

fig = plot_reward(logs[1][:rewards])
plot_rewards( map(x->x[:rewards], logs[1:3]) )

push!(logs, Dict(:rewards=>map(x->[DPMBIRL.RewardFunction(rand(100))], 1:500) ) )

covariances = [cov(rewards[:,i]) for i in 1:100]
Plots.plot(covariances)

find(x->x>0.00005, covariances)


using JLD
load("$(pwd())/MH_test.jld")



"""
	Given an array of logs, makes a summary table of
	- seed, initial τ, burn-in, number of agents, number of trajectories
	- reward mean
	- reward variance
	- reward median
	- acceptance rate mean
	= acceptance
	= likelihood
"""
function summary_statistics(_log, parameters, save_to)


	reward = log2reward(_log[:rewards])
	reward_mean = mean(reward,1)
	reward_median = median(reward,1)
	reward_variance = var(reward,1)

	acc_rate_mean = mean(_log[:acc_prob])

	summary = Dict( :reward_means => reward_mean,
					:reward_medians => reward_median,
					:reward_variance => reward_variance,
					:acc_rate_mean => acc_rate_mean
					)

	f = jldopen("$save_to/summary.jld", "r+")
	write(f, "summary", summary)
	close(f)
	seed = parameters["seed"]
	plot_reward(_log[:rewards], path_to_save = "$save_to/reward-$seed.png" )
	@show summary
end

function prepare_log_folder(save_to, parameters)
	minute, hour = Dates.minute(now()), Dates.hour(now())
	mkdir("$save_to/$hour-$minute")
	open("$save_to/$hour-$minute/setup.txt", "w+") do f
		for key in keys(parameters)
			write(f, "$key: $(parameters[key])\n")
		end
	end
	f = jldopen("$save_to/summary.jld", "w")
	close(f)

	"$save_to/$hour-$minute"
end
