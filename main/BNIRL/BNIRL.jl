# ASCII comments generated using http://patorjk.com Calvin S font
module BNIRL

using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox

export main, traj2obs, Observation

include("../DPM_BIRL/DPM_BIRL.jl")
include("helper.jl")


function main(mdp, observations, η, κ; seed=1, max_iter=5e4, burn_in=500, use_assignements=true, ground_truth=nothing, punishment=5)
	srand(seed)

	if !use_assignements
		@assert ground_truth !== nothing "Ground truth not given"
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
	const glb = Globals(n_states, n_actions, n_observations, support_space,
					n_support_states, ψ, state2goal, all_goals)

	if use_assignements
		goals = [sample(Goal, glb) for i in 1:3]
		z = rand([1,2,3], n_observations)
	else
		z = ground_truth
		n_goals = size(unique(z),1)
		goals =  [sample(Goal, glb) for i in 1:n_goals]
	end

	_log = Dict(:z=>[], :goals=>[])
	goal_hist = zeros(Integer, max_iter, 3)
	for t in 1:max_iter
		t%100 == 0 ? println("Iteration $t") : nothing

		# ╔═╗╔═╗╔═╗╦╔═╗╔╗╔  ╔═╗╔═╗╔═╗╦  ╔═╗
		# ╠═╣╚═╗╚═╗║║ ╦║║║  ║ ╦║ ║╠═╣║  ╚═╗
		# ╩ ╩╚═╝╚═╝╩╚═╝╝╚╝  ╚═╝╚═╝╩ ╩╩═╝╚═╝

		for (i,curr_goal) in enumerate(goals)
			# In this algorithm, the current goal apparently
			# has no "say" in the next sampled goal. Maybe could add
			# a term for that

			# Find the observations assigned to the current goal
			assigned_to_goal = (z .== i)

			# Calculate likelihood of observations given a goal
			goal_observations = observations[assigned_to_goal]
			probs_vector = likelihood_vector(goal_observations, goals, η, glb)

			# Use likelihoods to make a probability vector
			probs_vector /= sum(probs_vector)

			# Pick index and its related state
			chosen 		  = rand(Multinomial(1,probs_vector))
			state_chosen  = support_space[findfirst(chosen)]

			# Get the state from the goal
			goal_chosen = state2goal[state_chosen]
			goals[i]    = goal_chosen
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
		for (i,obs) in enumerate(observations)
			# Get the CRP probabilities
			CRP_probs = CRP(z, κ)
			llh_probs = zeros(size(goals,1)+1)

			# size should be number of current goals + 1
			# @show size(CRP_probs,1), size(llh_probs,1)
			# @assert size(CRP_probs,1) == size(llh_probs,1)

			# Sample a potential new goal
			potential_g = sample(Goal, glb)

			# Calculate likelihood of observation per goal
			for (j,g) in enumerate(goals)
				llh_probs[j] = likelihood(obs, g, η)
			end
			llh_probs[end] = likelihood(obs, potential_g, η)

			# Put probabilities together and normalise
			probs_vector  = llh_probs .* CRP_probs
			probs_vector /= sum(probs_vector)

			# Sample new assignement
			chosen = findfirst(rand(Multinomial(1,probs_vector)))
			z[i] = chosen
			if chosen == size(goals,1)+1
				push!(goals, potential_g)
				# info("Pushed cluster")
			end

			# ╔═╗┌─┐┌─┐┌┬┐  ╔═╗┬─┐┌─┐┌─┐┌─┐┌─┐┌─┐
			# ╠═╝│ │└─┐ │───╠═╝├┬┘│ ││  ├┤ └─┐└─┐
			# ╩  └─┘└─┘ ┴   ╩  ┴└─└─┘└─┘└─┘└─┘└─┘

			# Remove empty assignements and their goals
			tally_z = tally(z)
			# @show tally_z
			for i in reverse(1:size(tally_z,1))
				if tally_z[i] == 0
					z[ z .> i ] -= 1
					deleteat!(goals, i)
					# info("Deleted partition $i")
				end
			end
			if sum(iszero.(tally(z))) != 0
				@show tally(z)
				error("Size not conserved")
			end
			if size(tally(z),1) != size(goals,1)
				# The only explanation after thorough review is that
				# the last cluster went from 1 to 0 observations assigned
				# and therefore was out of reach of the tally which stops at the highest
				# cluster. Therefore, simply remove the last goal
				# TODO: do this, but less hacky
				pop!(goals)
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
