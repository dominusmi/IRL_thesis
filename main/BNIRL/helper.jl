import StatsBase: sample
import Base: ==, hash, isequal, copy, show

# TODO: Check that two observation with same state, diff actions are not "equal"
immutable Observation
	state
	action
end
hash(a::Observation, h::UInt) 			= hash(a.s, hash(a.a, hash(:Observation, h)))
isequal(a::Observation, b::Observation) = Base.isequal(hash(a), hash(b))
copy(o::Observation) 					= Observation(copy(o.s),copy(o.a))

immutable Goal
	state::Integer
	Q::Matrix{Float64}
end
==(g1::Goal,g2::Goal) = g1.state == g2.state
get_state(g::Goal) = g.state
show(io::IO, g::Goal) = print(io, "Goal:$(get_state(g))")

"""
	Returns an array states which are part of the observations
"""
function getSupportSpace(observations::Array{Observation})
	support = Array{Int64}(0)
	for obs in observations
		push!(support, obs.state)
	end
	unique(support)
end

"""
	Calculates the likelihood of an observation given a goal
"""
function likelihood(oᵢ::Observation, g::Goal, η )
	if g.state == oᵢ.state
		return 0.01
	end
	β = exp.( η * g.Q[oᵢ.state, :] )
	# exp( η * g.Q[oᵢ.state, oᵢ.action] )
	tmp = β[oᵢ.action] * (1 - 5*(maximum(β) - β[oᵢ.action]))

	tmp > 0.1 ? tmp : 0.1
end

"""
	Calculates the likelihood of several observations given a goal
"""
function likelihood_vector(observations::Vector{Observation}, goals::Vector{Goal}, η::AbstractFloat)
	global support_space, state2goal
	llh_vector = zeros(size(support_space,1))
	for obs in observations
		for (sᵢ, state) in enumerate(support_space)
			goal = state2goal[state]
			llh_vector[sᵢ] += likelihood(obs, goal, η)
		end
	end
	llh_vector
end


"""
	Transforms and MDPHistory array into an array of obsevations
"""
function traj2obs(mdp, trajectories::Array{MDPHistory})
	observations = Array{Observation}(0)
	for trajectory in trajectories
		obs = traj2obs(mdp, trajectory)
		push!(observations,obs...)
	end
	unique(observations)
end

function traj2obs(mdp, trajectory::MDPHistory)
	observations = Array{Observation}(0)
	for (h,state) in enumerate(trajectory.state_hist[1:end-1])
		aⁱ = action_index(mdp, trajectory.action_hist[h])
		sⁱ = state_index(mdp, state)
		obs = Observation(sⁱ, aⁱ)
		push!(observations,obs)
	end
	# unique(observations)
	observations
end

"""
	Returns a random goal from the support space
"""
function sample(::Type{Goal})
	global support_space, state2goal
	state = rand(support_space)
	state2goal[state]
end

"""
	Calculates the loss given a partitioning of the observations
"""
function partitioning_loss(goals, observations, z)
	loss = 0
	for (i,obs) in enumerate(observations)
		goal = goals[z[i]]
		policy_action = indmax(goal.Q[obs.state,:])
		if policy_action !== obs.action
			loss += 1
		end
	end
	loss
end


"""
	Precomputes Q values for every state in support space
"""
function precomputeQ(mdp, support_space)
	n_support_states = size(support_space,1)
	tmp_array= Array{Goal}(n_support_states)
	tmp_dict = Dict()
	utils = []
	# Solves mdp for each value
	for (i,state) in enumerate(support_space)
		# Prepare mdp with reward at state
		raw_mdp = copy(mdp)
		victory_state_idx = state
		victory_state = GridWorldState(DPMBIRL.i2s(mdp,state)...)
		# Remove any other positive reward
		idx = find(x->x>0., raw_mdp.reward_values)
		raw_mdp.reward_values[idx] = 0.
		# Set current state as positive reward and terminal
		raw_mdp.reward_values[victory_state_idx] = 1.
		raw_mdp.terminals = Set([victory_state])
		# Solve mdp
		state_policy = DPMBIRL.solve_mdp(raw_mdp)
		tmp_dict[state] = Goal(state, state_policy.qmat[1:end-1,:])
		tmp_array[i] = tmp_dict[state]
		push!(utils, state_policy.util[1:end-1])
	end
	tmp_dict, tmp_array, utils
end

"""
	Given an array, finds the number of times the array elements occur
"""
function tally(zd)
    ret = zeros(Int64, maximum(zd))
    for k in zd
        ret[k] += 1
    end
    return ret
end

"""
	Given a vector of assignements, calculates the CRP probability vector,
	setting the last element as the probability of instantiating a new cluster
"""
function CRP(assignements::Vector{<:Integer}, κ)
	occurences 	 = tally(assignements)
	_sum 		 = sum(occurences)
	denom 		 = _sum-1+κ
	probs_vector = zeros(size(occurences,1)+1)
	for i in 1:size(occurences,1)
		probs_vector[i] = occurences[i] / denom
	end
	probs_vector[end] = κ / denom
	probs_vector
end
