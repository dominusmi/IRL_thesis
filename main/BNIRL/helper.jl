import StatsBase: sample
import Base: ==, hash, isequal, copy, show, deleteat!
using Plots

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

immutable Globals
	n_states::Integer
	n_actions::Integer
	support_space::Array{<:Integer}
	n_support_states::Integer
	ψ::AbstractFloat
	state2goal::Dict
	all_goals::Array{Goal}
	η::AbstractFloat
	κ::AbstractFloat
end

mutable struct Clusters
	K::Integer
	N::Array{<:Integer}
	assignements::Array{<:Integer}
	G::Array{Goal}
	Z::Array{Any}
	ids::Array{<:Integer}
end

function copy(c::Clusters)
	Clusters(c.K, copy(c.N), copy(c.assignements), copy.(c.G), copy.(c.Z), copy(c.ids))
end

"""
	Returns an array states which are part of the observations
"""
function getSupportSpace(observations::Vector{Observation})
	support = Array{Int64}(0)
	for obs in observations
		push!(support, obs.state)
	end
	unique(support)
end

"""
	Calculates the likelihood of an observation given a goal
"""
function likelihood(oᵢ::Observation, g::Goal, glb::Globals)
	ψ, η = glb.ψ, glb.η
	if g.state == oᵢ.state
		return 0.01
	end
	β = exp.( η * g.Q[oᵢ.state, :] )
	# β = η * g.Q[oᵢ.state, :]
	tmp = β[oᵢ.action] * (1 - ψ*(maximum(β) - β[oᵢ.action]))

	tmp > 0.01 ? tmp : 0.01
end

"""
	Calculates the likelihood of several observations given a goal
"""
function likelihood_vector(observations::Vector{Observation}, goals::Vector{Goal}, glb::Globals)
	support_space, state2goal, η = glb.support_space, glb.state2goal, glb.η
	llh_vector = zeros(size(support_space,1))
	for obs in observations
		for (sᵢ, state) in enumerate(support_space)
			goal = state2goal[state]
			llh_vector[sᵢ] += likelihood(obs, goal, glb)
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
function sample(::Type{Goal}, glb::Globals)
	support_space, state2goal = glb.support_space, glb.state2goal
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
function CRP(assignements::Vector{<:Integer}, κ; use_clusters=true)
	occurences 	 = tally(assignements)
	_sum 		 = sum(occurences)
	if use_clusters
		denom = _sum-1+κ
		probs_size = size(occurences,1)+1
	else
		denom = _sum-1
		probs_size = size(occurences,1)
	end
	probs_vector = zeros( probs_size )
	for i in 1:size(occurences,1)
		probs_vector[i] = occurences[i] / denom
	end
	use_clusters ? probs_vector[end] = κ / denom : nothing
	probs_vector
end


function CRP(c::Clusters)
	if iszero(fixed_clusters)
        αs = zeros(c.K+1)
    else
        αs = zeros(c.K)
    end
    ∑N = size(c.assignements,1)
    cₘ = c.assignements[m]
    for k in 1:c.K
        Nₖ = c.N[k]
        if k == cₘ
            Nₖ -= 1
            Nₖ = (Nₖ == 0 ? 1e-5 : Nₖ)  # Dir doesn't like 0, so we set something very small
        end
        αs[k] = Nₖ / (α-1+∑N)
    end
    if iszero(fixed_clusters)
        αs[end] = α / (α-1+∑N)
    end
    indmax(rand(Dirichlet(αs),1))
end

function resample(goals::Array{Goal}, goal_idx, z, observations, glb)
	# In this algorithm, the current goal apparently
	# has no "say" in the next sampled goal. Maybe could add
	# a term for that

	# Find the observations assigned to the current goal
	assigned_to_goal = (z .== goal_idx)

	# Calculate likelihood of observations given a goal
	goal_observations = observations[assigned_to_goal]
	probs_vector = likelihood_vector(goal_observations, goals, glb)

	# Use likelihoods to make a probability vector
	probs_vector /= sum(probs_vector)

	# Pick index and its related state
	chosen 		  = rand(Multinomial(1,probs_vector))
	state_chosen  = glb.support_space[findfirst(chosen)]
	goal_chosen   = glb.state2goal[state_chosen]
end

function reassign!(obs::Observation, obs_idx, z, goals, glb; use_clusters=true)
	# Get the CRP probabilities
	CRP_probs = CRP(z, glb.κ, use_clusters=use_clusters)
	probs_size = use_clusters ? size(goals,1)+1 : size(goals,1)
	llh_probs = zeros(probs_size)

	# Sample a potential new goal
	potential_g = sample(Goal, glb)

	# Calculate likelihood of observation per goal
	for (j,g) in enumerate(goals)
		llh_probs[j] = likelihood(obs, g, glb)
	end

	use_clusters ? llh_probs[end] = likelihood(obs, potential_g, glb) : nothing

	# Put probabilities together and normalise
	probs_vector  = llh_probs .* CRP_probs
	probs_vector /= sum(probs_vector)

	# Sample new assignement
	chosen = findfirst(rand(Multinomial(1,probs_vector)))
	z[obs_idx] = chosen
	if chosen == size(goals,1)+1 && use_clusters
		push!(goals, potential_g)
	end
end


function update_cluster!(clusters, m, glb)
	new_cluster = false
	cₘ   = clusters.assignements[m] # Current assignement
	cₘ⁻  = sample(clusters,m,κ,fixed_clusters)     # Potential new assignement

	if cₘ⁻ == cₘ
		return
	elseif cₘ⁻ == clusters.K+1
		# If new cluster, sample new goals, conserve number of goals
		n_goals = size( unique(clusters.assignements[cₘ]),1 )
		gs⁻ = [sample(Goal, glb) for i in 1:n_goals]
		new_cluster = true
	else
		# Otherwise "load" goal
		g⁻ = glb.all_goals[cₘ⁻]
	end

	# Calculate likelihood
	𝓛, 𝓛⁻ = 0., 0.
	for (i, obs) in enumerate(trajectory[m])
		curr_ass = clusters.Z[cₘ][i]
		curr_goal = clusters.G[cₘ][curr_ass]
		𝓛  += likelihood(obs, curr_goal, glb)
		𝓛⁻ += likelihood(obs, gs⁻[curr_ass], glb)
	end
	accept = accept_proposition(Likelihood, 𝓛⁻, 𝓛)

	# Update if accepted
	if accept
		changes += 1
		# Update cluster and reward assignements
		if new_cluster
			# Add new cluster
			push!(clusters.G, gs⁻)
			push!(clusters.Z, copy.(clusters.Z[cₘ]))
			push!(clusters.N, 1)
			push!(clusters.ids, clusters.K+1)
			clusters.K += 1
			clusters.N[cₘ] -= 1
			clusters.assignements[m] = cₘ⁻
		else
			# Update clusters to new assignement
			clusters.N[cₘ] -= 1
			clusters.N[cₘ⁻] += 1
			clusters.assignements[m] = cₘ⁻
			push!(updated_clusters_id, clusters.ids[cₘ], clusters.ids[cₘ⁻])
		end
	end
	# Remove empty clusters
	if (to_remove = findfirst(x->x==0, clusters.N))>0
		deleteat!(clusters, to_remove)
		if to_remove ∈ updated_clusters_id
			delete!(updated_clusters_id,to_remove)
		end
	end
end


function postprocess!(z, goals)
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
	if size(tally(z),1) != size(goals,1)
		# The only explanation after thorough review is that
		# the last cluster went from 1 to 0 observations assigned
		# and therefore was out of reach of the tally which stops at the highest
		# cluster. Therefore, simply remove the last goal
		# TODO: do this, but less hacky
		pop!(goals)
	end
end


function deleteat!(c::Clusters, index::Integer)
    deleteat!(c.N, index)
    deleteat!(c.Z, index)
	deleteat!(c.G, index)
    deleteat!(c.ids, index)
    c.K -= 1
    temp = c.assignements .> index
    c.assignements[temp] -= 1
    return
end
