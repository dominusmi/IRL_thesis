using JLD
using Plots

# results_folder = "/home/edoardo/Documents/Masters Work/Dissertation/results/DPM_BIRL"
# results_folder = "/run/user/1000/gvfs/sftp:host=godzilla.csc.warwick.ac.uk,user=phujsc/home/space/phujsc/Documents/main/results"
#
# results = load("$results_folder/2_40.jld")
# results = results["results"]
#
# lhs = get_lhs(results, 1.0)
#
# evds = get_EVD_matrix(results, 1.0)
# lhs  = _log[:likelihoods]
# evds_1 = zeros(350, 4)
# evds_2 = zeros(350, 4)
#
# lh_1 = zeros(350, 4)
#
# for (i,evd) in enumerate(get_EVD_matrix(results, 1.0))
# 	for j in 1:4
# 		if j <= size(evd,2)
# 			evds_1[i,j] = evd[1,j]
# 			evds_2[i,j] = evd[2,j]
# 			lh_1[i,j] = lhs[i][j]
# 		end
# 	end
# end
#
# fig = Plots.plot(evds_1[:,1])
# Plots.plot!(lh_1[:,1]/maximum(lh_1[:,1])*20)
# savefig(fig, "MCMC 400 iterations")

#### Acceptance Probability ####
function plot_att_prob(_log)
	temp = abs.(_log[:acc_prob]) .< 3.
	acc_prob = _log[:acc_prob][temp]
	# temp = abs.(_log[:acc_prob]) .< 1e-3
	fig = Plots.plot(acc_prob, ylim=(-.1,2.),
				title="Acceptance probabilities using Langevin update\nNo prior, normalised reward",
				xlabel="Iterations",
				ylabel="P")
	# savefig("langevin acc prob - prior, norm-reward.png")
	fig
end

#### Likelihoods ####
function plot_likelihood(_log)
	fig = Plots.plot([ _log[:likelihoods][i][1] for i in 1:size(_log[:likelihoods],1)],
				title="Likelihoods using Langevin update",
				xlabel="Iteration",
				ylabel="Likelihood")
	# savefig("langevin likelihood - prior, norm reward.png")
	fig
end

#### Change rate ####
function plot_change(_log)
	fig = Plots.plot([ _log[:changed][i][1] for i in 1:size(_log[:changed],1)],
				title="Accepted Iterations",
				xlabel="Iteration",
				ylabel="Changed")
	fig
end

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
		println(path_to_save)
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
function summary_statistics(rewards, parameters; save_to=nothing)

	burn_in = parameters["Burn in"]
	reward = log2reward(rewards[burn_in:end])
	reward_mean = mean(reward,1)'
	reward_median = median(reward,1)'
	reward_variance = var(reward,1)'

	# acc_rate_mean = mean(_log[:acc_prob][burn_in:end])

	summary = Dict( :reward_means => reward_mean,
					:reward_medians => reward_median,
					:reward_variance => reward_variance,
					# :acc_rate_mean => acc_rate_mean
					)
	seed = (parameters["seed"])
	if save_to !== nothing
		f = jldopen("$save_to/summary-$seed.jld", "r+")
		write(f, "summary", summary)
		close(f)
	end
	seed = parameters["seed"]
	path_to_save = save_to
	summary, plot_reward(rewards, path_to_save = path_to_save )
end

function prepare_log_folder(save_to, parameters)
	minute, hour = Dates.minute(now()), Dates.hour(now())
	mkdir("$save_to")
	# open("$save_to/$hour-$minute/setup.txt", "w+") do f
		# for key in keys(parameters)
			# write(f, "$key: $(parameters[key])\n")
		# end
	# end

	"$save_to/"
end

function prepare_log_file(save_to, seed)
	f = jldopen("$save_to/summary-$seed.jld", "w")
	close(f)
end

"""
	Plots the reward function using different seed
"""
function plot_rewards_from_diff_seed(summaries)
	fig = Plots.plot(title="Rewards given different initial conditions", legend=false,
					xlabel="State", ylabel="Reward")
	for summary in summaries
		# test = load(pwd()*"/results/10-48/summary-$i.jld")
		summary = load(summary)["summary"]
		Plots.plot!(summary[:reward_means])
	end
	savefig(fig, "Result different initial conditions")
	# Plots.plot(test[:reward_medians], yerr=sqrt.(test[:reward_variance]))
#
#
# heatmap(reshape(mdps[1].reward_values, (10,10)))
# heatmap(reshape(test[:reward_means], (10,10)))
end



function get_EVD_matrix(results, confidence)
	for key in keys(results)
		if key == confidence
			return results[confidence][:EVDs]
		end
	end
end

function get_lhs(results, confidence)
	for key in keys(results)
		if key == confidence
			return results[confidence][:likelihoods]
		end
	end
end

function get_log(results, confidence)
	for key in keys(results)
		if key == confidence
			return results[confidence]
		end
	end
end

function final_number_partitions(results, confidence)
	for key in keys(results)
		if key == confidence
			return size(results[confidence][:EVDs],1)
		end
	end
end
