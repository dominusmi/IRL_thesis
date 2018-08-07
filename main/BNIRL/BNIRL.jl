using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox
import StatsBase: sample
import Base: ==

include("../DPM_BIRL/DPM_BIRL.jl")


immutable Observation
	s
	a
end
Base.hash(a::Observation, h::UInt) = hash(a.s, hash(a.a, hash(:Observation, h)))
Base.isequal(a::Observation, b::Observation) = Base.isequal(hash(a), hash(b))

immutable Goal
	state::Integer
	Q::Matrix{Float64}
end
==(g1::Goal,g2::Goal) = g1.state == g2.state

abstract type Prior end
immutable DiscretePrior <: Prior
	pdf::Array{<:AbstractFloat}
	prior::Distribution
end


mutable struct Partition
	g::Goal
	O::Array{Observation}
end


function getSupportSpace(mdp, trajectories::Array{MDPHistory}, n_states)
	support = zeros(Int64, n_states)
	for (i,trajectory) in enumerate(trajectories)
		support += getSupportSpace(mdp, trajectory, n_states)
	end
	support
end

function getSupportSpace(mdp, trajectory::MDPHistory, n_states)
	support = zeros(Int64, n_states)
	for state in trajectory.state_hist[1:end-1]
		support[state_index(mdp,state)] += 1
	end
	support
end

function likelihood(oᵢ::Observation, g::Goal, η )
	denom = sum(exp.( η * g.Q[:, oᵢ.a] ))
	exp( η * g.Q[oᵢ.s, oᵢ.a] ) / denom
end


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
	unique(observations)
end

function sample(::Type{Goal}, p::Prior, goals_dict)
	idx = findfirst(x->x!=0, rand(p.prior))
	goals_dict[idx]
end

prior(g::Goal, p::DiscretePrior) = p.pdf[g.state]

function tally(zd)
    ret = zeros(Int64, maximum(zd))
    for k in zd
        ret[k] += 1
    end
    return ret
end

function CRP(assignements::Vector{<:Integer}, κ)
	occurences = tally(assignements)
	_sum = sum(occurences)
	denom = _sum-1+κ
	probs_vector = zeros(size(occurences,1)+1)
	for i in 1:size(occurences,1)
		probs_vector[i] = occurences[i] / denom
	end
	probs_vector[end] = κ / denom
	probs_vector
end

### Initialise problem and generate trajectories
η, κ = 1.0, 1.0
mdp, policy = DPMBIRL.generate_gridworld(10,10,γ=0.9)
trajectories = DPMBIRL.generate_trajectories(mdp, policy, 10)
observations = traj2obs(mdp, trajectories)

### Precompute all Q-values and their πᵦ
n_states = size(states(mdp),1)-1
n_actions = size(actions(mdp),1)
n_observations = size(observations,1)
support_space = getSupportSpace(mdp, trajectories, n_states)

n_support_states = sum(support_space.>0)
tmp_array= Array{Int64}(n_support_states)
# goals_dict is a dictionary state→Goal(state)
goals_dict = Dict()

# Solves mdp for each value
for (i,state) in enumerate(find(x->x>0., support_space))
	raw_mdp = copy(mdp)
	victory_state_idx = state
	victory_state = GridWorldState(DPMBIRL.i2s(mdp,state)...)
	idx = find(x->x>0., raw_mdp.reward_values)
	raw_mdp.reward_values[idx] = 0.
	raw_mdp.reward_values[victory_state_idx] = 1.
	raw_mdp.terminals = Set([victory_state])
	state_policy = DPMBIRL.solve_mdp(raw_mdp)
	tmp_array[i] = state
	goals_dict[state] = Goal(state, state_policy.qmat[1:end-1,:])
end

const goals_array = tmp_array

# Find prior on support set
support_space_prior = support_space / sum(support_space)
goals_prior = DiscretePrior( support_space_prior, Multinomial(1, support_space_prior) )

### Sample initial subgoal
G = []
push!(G, sample(Goal, goals_prior, goals_dict))

### Initialise assignements zᵢ
Z = []
zᵗ = ones(Integer, n_observations)
push!(Z, zᵗ)

### Initialiase partitions
partitions = Array{Partition}(0)
push!(partitions, Partition(G[1], observations))

next_g = []
next_z = zeros(Int64,n_observations)


# Find prior on support set
support_space_prior = support_space / sum(support_space)
goals_prior = DiscretePrior( support_space_prior, Multinomial(1, support_space_prior) )

### Sample initial subgoal
G = []
push!(G, sample(Goal, goals_prior, goals_dict))

### Initialise assignements zᵢ
Z = []
zᵗ = ones(Integer, n_observations)
push!(Z, zᵗ)

### Initialiase partitions
partitions = Array{Partition}(0)
push!(partitions, Partition(G[1], observations))

next_g = []
next_z = zeros(Int64,n_observations)
set_goals = Set{Goal}(values(goals_dict))
current_goals = Set([G[1]])
### Main loop
for t in 1:1
	zᵗ = Z[t]
	gᵗ = G[t]
	for p in partitions
		gᵗ = p.g
		prob_vector = zeros(n_support_states)
		# Calculate likelihood for all possible subgoals
		for (i,stateᵢ) in enumerate(goals_array)
			gⱼ = goals_dict[stateᵢ]
			llh = 0.
			for oᵢ in p.O
				llh += likelihood(oᵢ,gⱼ, η)
			end
			prob_vector[i] = llh * prior(gⱼ, goals_prior)
		end
		# Sample from multinomial
		s = rand( Multinomial(1, prob_vector / sum(prob_vector)) )
		s = findfirst(x->x==1,s)
		@show s
		break
		# Set new goal
		push!(next_g, collect(keys(goals_dict))[s])
	end

	@show next_g
	next_z = zeros(Int64,n_observations)
	CRP_vector = CRP(zᵗ, κ)
	new_partitions = []
	for (i,oᵢ) in enumerate(observations)
		probs_vector = copy(CRP_vector)
		for (j,p) in enumerate(partitions)
			# j: cluster of g
			### Calculate p(zᵢ=j|z,O,Rⱼ)
			# P(zᵢ|z₋ᵢ) = #observations in zᵢ / n-1+η
			# P(oᵢ|g_zᵢ) = likelihood of observation given assignement
			probs_vector[j] *= likelihood(oᵢ, p.g, η)
		end
		# New subgoal
		new_g = sample(Goal, goals_prior, goals_dict)
		if new_g ∈ current_goals
			# If sampled subgoal already in use, find in which partition
			chosen = findfirst(x->x.g == new_g, partitions)
		else
			# Otherwise, prepare for new partition
			probs_vector[end] *= likelihood(oᵢ, new_g, η)

			# Normalise to get probabilities
			probs_vector /= sum(probs_vector)
			@show probs_vector
			# Sample multinomial
			s = rand( Multinomial(1,probs_vector) )
			chosen = findfirst(x->x==1, s)

			# Prepare for new partitions or set new assignement
			if chosen == size(probs_vector,1)
				push!(new_partitions, [i, new_g])
			end
		end

		# Set new assignement
		next_z[i] = chosen
	end
	# Add/remove partitions
	post_process!(partitions, observations, zᵗ, next_z, new_partitions)

	push!(Z, next_z)
	push!(G, next_g)
end

function post_process!(partitions,  observations, old_z, new_z, new_partitions)
	new_ass_idx = size(partitions,1)+1

	# Find assignements that changed
	changed = old_z .!= new_z
	# Add new partitions if necessary
	new_goals = Set(map(x->x[2], new_partitions))
	# Loop through goals
	for g in new_goals
		temp_ass = Array{Observation}(0)
		# Finds observations corresponding to new goal
		for np in new_partitions
			if np[2] == g
				# Add them to temporary assignements
				push!(temp_ass, observations[np[1]])
			end
		end
		push!(partitions, Partition(g, temp_ass))
	end

	# Re-assign observations
	for (obs_idx, p_idx) in enumerate(new_z[changed])
		if p_idx != new_ass_idx
			# Find old partition
			old_partition_id = old_z[obs_index]
			# Get observation
			obs = observation[obs_idx]
			# Remove obs from old partition
			filter!(x->x==obs, partitions[ old_partition_id ] )
			# Add observation to new partition
			push!(partitions[p_idx], observations[obs_idx])
		end
	end
end
