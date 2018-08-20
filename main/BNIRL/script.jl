using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox
using Plots
include("../DPM_BIRL/DPM_BIRL.jl")
include("BNIRL.jl")
# reload("BNIRL")
include("BNIRL.jl")
using BNIRL
pyplot()

### Initialise problem and generate trajectories
# anim = @animate for γ in 0.4:0.05:0.99
# anim = @animate for punishment in 10*(2:2:8)
	γ = 0.95
	println("γ: $γ")
	srand(5)
	η, κ = 1.0, 0.1
	mdp, policy = DPMBIRL.generate_diaggridworld(10,10,γ=γ)
	# mdp.reward_states = mdp.reward_states[mdp.reward_values .> 0.]
	# mdp.reward_values = [mdp.reward_values[i] > 0. ? 1.0 : 0.0 for i in 1:100]
	# punishment = exp(inv(1-γ^2))
	punishment = inv(1-γ^2)
	trajectories, z = DPMBIRL.generate_subgoals_trajectories(mdp, GridWorldState(2,1), [GridWorldState(1,8), GridWorldState(10,10)])
	observations = BNIRL.traj2obs(mdp, trajectories)

	_log, glb = BNIRL.main(mdp, observations, η, κ; max_iter=5_000, burn_in=2_000, use_assignements=true, ground_truth=z, punishment=punishment, use_clusters=true, n_goals=2)

	plot_partition_sizes(_log[:goals])
	n_goals = 3
	fig_g, fig_z = [], []
	for i in 1:n_goals
		push!(fig_g, goals_plot(_log[:goals],n_goals,i, glb))
		# fig_2 = goals_plot(_log[:goals],3,2, glb)
		# fig_3 = goals_plot(_log[:goals],3,3, glb)

		push!(fig_z, z_plot(_log[:z],n_goals,i))
		# fig_z2 = z_plot(_log[:z],3,2)
		# fig_z3 = z_plot(_log[:z],3,3)
	end

	fig = Plots.plot(fig_g...,fig_z..., layout=(2,n_goals))
# end




gif(anim, fps=2)

function plot_partition_sizes(goals)
	sizes = [ size(goals[i],1) for i in 1:size(goals,1) ]
	histogram(sizes)
end

function goals_plot(goals, n_goals, goal_id, glb)
	indeces = find( x->size(x,1)==n_goals, goals)

	objs = zeros(size(indeces,1),n_goals)
	[ objs[i,:] = goals[index] for (i, index) in enumerate(indeces)]


	vector = []
	for s in glb.support_space
		push!(vector, count(objs[:,goal_id].==s))
	end
	bar(glb.support_space, vector, legend=false, xticks=0:10:101, bar_width=1.)
end


function z_plot(z, n_goals, goal_id)
	n_obs = size(z[1],1)
	indeces = find( x->size(unique(x),1)==n_goals, z)
	objs = zeros(size(indeces,1),n_obs)

	for (i,ass) in enumerate(z[indeces])
		objs[i,find(ass.==goal_id)] += 1
	end
	_sum = sum(objs,1)
	bar(collect(1:n_obs), _sum[1,:])
end

srand(5)
γ = 0.5
η, κ = 1.0, 1.0
mdp, policy = DPMBIRL.generate_gridworld(10,10,γ=γ)
trajectories, z = DPMBIRL.generate_subgoals_trajectories(mdp, GridWorldState(2,5), [GridWorldState(2,4), GridWorldState(2,6)])
observations = BNIRL.traj2obs(mdp, trajectories)
support_space = BNIRL.getSupportSpace(observations)
_goals, _, _utils = BNIRL.precomputeQ(mdp, support_space)
_goals[52].Q[42,:]




heatmap(reshape(mdp.reward_values,(10,10)))
heatmap(reshape(_utils[1],(10,10)))


_goals


function plot_observations(mdp,	O; colours=nothing)
	fig = Plots.plot([], [], xlim=(0.5, mdp.size_x+0.5), ylim=(0.5, mdp.size_y+0.5),
							xticks=1:mdp.size_x, yticks=1:mdp.size_y)

	n_obs = size(O,1)
	for i in 1:n_obs-1
		s1, s2 = DPMBIRL.i2s(mdp, O[i].state), DPMBIRL.i2s(mdp, O[i+1].state)
		x = [s1[1], s2[1]]
		y = [s1[2], s2[2]]
		Plots.plot!(x,y, arrow=true, legend=false)
	end
	fig
end
