using Plots, JLD
using POMDPs
using POMDPModels

n_clusters_posterior = map(x->x.K, _log[:clusters])
Plots.plot(n_clusters_posterior)
mean(n_clusters_posterior)

""" Does tally of cluster posterior """
t = Dict([(i,count(x->x==i,n_clusters_posterior)) for i in unique(n_clusters_posterior)])

summary = summary_statistics(_log, parameters)
extr_reward = summary[:rewards_posterior][2][:summaries][2][:reward_means]
extr_reward = summary[:rewards_posterior][2][:figs][2]

""" Plot heatmap of gridworld and reward """
plot_mdp(mdp::MDP) = heatmap(reshape(mdp.reward_values,(10,10)), legend=false)
plot_mdp(mdp_rewards::Array{<:AbstractFloat}) = heatmap(reshape(mdp_rewards,(10,10)), legend=false)
plot_r(r) = heatmap(reshape(r,(10,10)), legend=false)

""" Plots layed out heatmap of mdps and rewards, k should be ground truth """
function truth_comparison_plot(mdps, summary, k)
	mdp_figs = Array{Any}(k)
	r_figs = Array{Any}(k)

	for i in 1:k
		mdp_figs[i] = plot_mdp(mdps[i])
		r = summary[:rewards_posterior][k][:summaries][i][:reward_means]
		r_figs[i] = plot_r(r)
	end
	Plots.plot(mdp_figs..., r_figs..., layout=(2,k))
end
fig = truth_comparison_plot(mdps, summary, 2)

folder = "results/results 2000 iterations/"

using Glob
for file in glob(folder*"summary_*_*_*.jld")
	filename = split(file,"/")[end]
	n_agents = split(file,"_")[3]
	summary, ground_truth = load(file)
	# Loaded as pairs of (string => value)
	summary, ground_truth = summary[2], ground_truth[2]

	# plot_mdp(ground_truth[2])
	@show filename
	try
		truth_comparison_plot(ground_truth, summary, parse(n_agents))
		savefig(file*".png")
	catch KeyError
		warn("Could not find posterior for reward $n_agents in $filename")
	end
end


load(folder*"summary_1_3_40.jld")
