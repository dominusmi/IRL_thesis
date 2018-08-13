# ASCII comments generated using http://patorjk.com Calvin S font
module BNIRL

using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox

export main, traj2obs, Observation

include("../DPM_BIRL/DPM_BIRL.jl")
include("helper.jl")


function main(mdp, observations, η, κ; seed=1, max_iter=5e4, burn_in=500)
	srand(seed)


	# Setup general variables
	global const n_states 			= size(states(mdp),1)-1
	global const n_actions 			= size(actions(mdp),1)
	global const n_observations 	= size(observations,1)
	global const support_space 		= getSupportSpace(observations)
	global const n_support_states 	= size(support_space,1)

	### Precompute all Q-values and their πᵦ
	tmp_dict, tmp_array, utils = precomputeQ(mdp, support_space)
	global const state2goal = tmp_dict
	global const all_goals  = tmp_array

	goals = [sample(Goal) for i in 1:3]
	z = rand([1,2,3], n_observations)


	_log = Dict(:z=>[], :goals=>[])
	goal_hist = zeros(Integer, max_iter, 3)
	for t in 1:max_iter
		# Re-sample goals
		t%10 == 0 ? println("Iteration $t") : nothing
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
			probs_vector = likelihood_vector(goal_observations, goals, η)

			# Use likelihoods to make a probability vector
			probs_vector /= sum(probs_vector)

			# Pick index and its related state
			chosen 		  = rand(Multinomial(1,probs_vector))
			state_chosen  = support_space[findfirst(chosen)]

			# Get the state from the goal
			goals[i] 	  = state2goal[state_chosen]
		end

		# @show get_state.(goals)
		# push!(_log, partitioning_loss(goals, observations, z))

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
			potential_g = sample(Goal)

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
	_log
end

end
