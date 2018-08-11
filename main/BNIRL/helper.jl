# TODO: Check that two observation with same state, diff actions are not "equal"
immutable Observation
	state
	action
end
Base.hash(a::Observation, h::UInt) = hash(a.s, hash(a.a, hash(:Observation, h)))
Base.isequal(a::Observation, b::Observation) = Base.isequal(hash(a), hash(b))
Base.copy(o::Observation) = Observation(copy(o.s),copy(o.a))

immutable Goal
	state::Integer
	Q::Matrix{Float64}
end
==(g1::Goal,g2::Goal) = g1.state == g2.state
get_state(g::Goal) = g.state
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
		return 0.
	end
	denom = sum(exp.( η * g.Q[oᵢ.state, :] ))
	exp( η * g.Q[oᵢ.state, oᵢ.action] ) / denom
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
	end
	tmp_dict, tmp_array
end
