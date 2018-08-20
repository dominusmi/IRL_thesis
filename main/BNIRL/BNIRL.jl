# ASCII comments generated using http://patorjk.com Calvin S font
module BNIRL

using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox

export main, traj2obs, Observation

include("../DPM_BIRL/DPM_BIRL.jl")
include("helper.jl")
include("MABNIRL.jl")

function main(mdp, observations, η, κ; seed=1, max_iter=5e4, burn_in=500, use_assignements=true, ground_truth=nothing, punishment=5, use_clusters=false, n_goals=0 )
	srand(seed)

	if !use_assignements
		@assert ground_truth !== nothing "Ground truth not given"
	end

	if !use_clusters
		@assert n_goals !== 0 "Must give a number of clusters"
	end



	n_states 			= size(states(mdp),1)-1
	n_actions 			= size(actions(mdp),1)
	n_observations 		= size(observations,1)
	support_space 		= getSupportSpace(observations)
	n_support_states 	= size(support_space,1)
	ψ					= punishment

	println("Punishment: $ψ")

	### Precompute all Q-values and their πᵦ
	tmp_dict, tmp_array, utils = precomputeQ(mdp, support_space)
	state2goal = tmp_dict
	all_goals  = tmp_array

	# Setup general variables
	const glb = Globals(n_states, n_actions, support_space,
					n_support_states, ψ, state2goal, all_goals, η, κ)

	if !use_assignements
		z = ground_truth
		n_goals = size(unique(z),1)
		goals =  [sample(Goal, glb) for i in 1:n_goals]
	elseif !use_clusters
		z = rand(1:n_goals, n_observations)
		goals =  [sample(Goal, glb) for i in 1:n_goals]
	else
		goals = [sample(Goal, glb) for i in 1:3]
		z = rand([1,2,3], n_observations)
	end

	_log = Dict(:z=>[], :goals=>[])
	goal_hist = zeros(Integer, max_iter, 3)
	for t in 1:max_iter
		t%100 == 0 ? println("Iteration $t") : nothing

		# ╔═╗╔═╗╔═╗╦╔═╗╔╗╔  ╔═╗╔═╗╔═╗╦  ╔═╗
		# ╠═╣╚═╗╚═╗║║ ╦║║║  ║ ╦║ ║╠═╣║  ╚═╗
		# ╩ ╩╚═╝╚═╝╩╚═╝╝╚╝  ╚═╝╚═╝╩ ╩╩═╝╚═╝

		for (i,curr_goal) in enumerate(goals)
			# Find the observations assigned to the current goal
			assigned_to_goal = (z .== goal_idx)

			# Calculate likelihood of observations given a goal
			goal_observations = observations[assigned_to_goal]

			goals[i] = resample(goals, goal_observations, glb)
		end

		if !use_assignements
			if t > burn_in
				push!(_log[:goals], get_state.(goals))
			end
			continue
		end

		# ╔═╗╔═╗╔═╗╦╔═╗╔╗╔  ╔═╗╔╗ ╔═╗╔═╗╦═╗╦  ╦╔═╗╔╦╗╦╔═╗╔╗╔╔═╗
		# ╠═╣╚═╗╚═╗║║ ╦║║║  ║ ║╠╩╗╚═╗║╣ ╠╦╝╚╗╔╝╠═╣ ║ ║║ ║║║║╚═╗
		# ╩ ╩╚═╝╚═╝╩╚═╝╝╚╝  ╚═╝╚═╝╚═╝╚═╝╩╚═ ╚╝ ╩ ╩ ╩ ╩╚═╝╝╚╝╚═╝

		# Re-assign observations
		tmp_use_clusters = use_clusters
		for rep in 1:5
		for (i,obs) in enumerate(observations)
			reassign!(obs, i, z, goals, glb, use_clusters=tmp_use_clusters)
			postprocess!(z, goals)

			if !use_clusters
				if size(goals,1) !== n_goals
					tmp_use_clusters = true
				else
					tmp_use_clusters = false
				end
			end
		end
		end

		if t>burn_in
			push!(_log[:z], copy(z))
			push!(_log[:goals], get_state.(goals))
		end
	end
	_log, glb
end

end
