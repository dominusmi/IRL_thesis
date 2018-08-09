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
Base.copy(o::Observation) = Observation(copy(o.s),copy(o.a))

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
	# unique(observations)
	observations
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

function calLoss(zᵗ, partitions, observations)
	loss = 0
	for (i,obs) in enumerate(observations)
		p_idx = zᵗ[i]
		p = partitions[p_idx]
		if indmax(p.g.Q[obs.s,:]) !== obs.a
			loss +=1
		end
	end
	loss
end

fig = @gif for obs in observations
	pos = DPMBIRL.i2s(mdp,obs.s)
	println(pos)
	scatter([pos[1]], [pos[2]], xlim=(0,10), ylim=(0,10))
end
fig

### Initialise problem and generate trajectories
srand(1)
η, κ = 1.0, 1.0
mdp, policy = DPMBIRL.generate_gridworld(10,10,γ=0.9)
# trajectories = DPMBIRL.generate_trajectories(mdp, policy, 10)
trajectories = DPMBIRL.generate_subgoals_trajectories(mdp)
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
function _test(;max_iter=10)
	G = []
	push!(G, sample(Goal, goals_prior, goals_dict))

	### Initialise assignements zᵢ
	Z = []
	zᵗ = ones(Integer, n_observations)
	push!(Z, zᵗ)

	### Initialiase partitions
	partitions = Array{Partition}(0)
	push!(partitions, Partition(G[1], copy.(observations)))


	set_goals = Set{Goal}(values(goals_dict))
	current_goals = Set([G[1]])
	### Main loop
	for t in 1:max_iter
		println("Iteration $t")

		# Set up iteration variables
		zᵗ = Z[t]
		gᵗ = G[t]

		# Calculate loss
		loss = calLoss(zᵗ, partitions, observations)
		@show loss

		# Update each subgoal
		for g in gᵗ
		# for p in partitions
			# g = p.g
			# For each subgoal, must compute the probability of the currently assigned
			# observations be of another goal
			probs_vector = zeros(n_support_states)
			# Calculate likelihood for all possible subgoals
			for (i,stateᵢ) in enumerate(goals_array)
				gⱼ = goals_dict[stateᵢ]
				llh = 0.
				for oᵢ in p.O
					llh += likelihood(oᵢ,gⱼ, η)
				end
				probs_vector[i] = llh #* prior(gⱼ, goals_prior)
			end
			# Sample from multinomial
			s = rand( Multinomial(1, probs_vector / sum(probs_vector)) )
			s = findfirst(x->x==1,s)

			# Acceptance Step
			if goals_array[s] ≠ g.state
				# Not likelihood since prior part of it
				curr_llh = probs_vector[findfirst(goals_array .== g.state)]
				prop_llh = probs_vector[s]
				accept = true
				# if prop_llh < curr_llh
				# 	accept = rand() < prop_llh / curr_llh
				# end
				if accept
					# Set new goal
					updated_goal = goals_dict[goals_array[s]]
					# p.g = updated_goal
					g = updated_goal
					# accepted_counter += 1
				end
			end
		end
		next_z = zeros(Int64,n_observations)

		next_g = map(x->x.g, partitions)
		current_goals = Set(next_g)

		CRP_vector = CRP(zᵗ, κ)
		_temp = map(x->x.g.state, partitions)
		new_partitions = []
		for (i,oᵢ) in enumerate(observations)
			probs_vector = copy(CRP_vector)
			for (j,g) in enumerate(gᵗ)
				# j: cluster of g
				### Calculate p(zᵢ=j|z,O,Rⱼ)
				# P(zᵢ|z₋ᵢ) = n_observations in zᵢ / n-1+η
				# P(oᵢ|g_zᵢ) = likelihood of observation given assignement
				probs_vector[j] *= likelihood(oᵢ, g, η)
			end
			# New subgoal
			prop_g = sample(Goal, goals_prior, goals_dict)
			if prop_g ∈ current_goals
				# If sampled subgoal already in use, find in which partition
				# @show new_g.state
				chosen = findfirst(gᵗ .== prop_g)
				# @show chosen
			else
				# Otherwise, prepare for proposed partition
				probs_vector[end] *= likelihood(oᵢ, prop_g, η)

				# Normalise to get probabilities
				probs_vector /= sum(probs_vector)
				# Sample multinomial
				s = rand( Multinomial(1,probs_vector) )
				chosen = findfirst(x->x==1, s)

				# Prepare for new partitions or set new assignement
				if chosen == size(probs_vector,1)
					push!(new_partitions, [i, prop_g])
				end
			end

			# Acceptance step
			if chosen ≠ zᵗ[i]
				curr_llh = probs_vector[zᵗ[i]]
				prop_llh = probs_vector[chosen]
				accept = true
				if prop_llh < curr_llh
					accept = rand() < prop_llh / curr_llh
				end
			end

			# Set new assignement
			if chosen == 0
				@show "Chosen = 0 at observation $i"
			end
			next_z[i] = chosen
		end

		# Add/remove partitions
		post_process!(partitions, observations, zᵗ, next_z, new_partitions, t)

		next_g = map(x->x.g, partitions)
		current_goals = Set(next_g)

		push!(Z, next_z)
		push!(G, next_g)
	end
	Z,G
end

srand(1)
Z,G = _test()



@gif for goals in G[end-100:end]
	pos = zeros(size(goals,1),2)
	for (i,goal) in enumerate(goals)
		tmp = DPMBIRL.i2s(mdp, goal.state)
		pos[i,:] = [tmp[1],tmp[2]]
	end
	scatter(pos[:,1], pos[:,2], xlim=(0,10), ylim=(0,10))
end


function post_process!(partitions,  observations, old_z, new_z, new_partitions, t)
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
				push!(temp_ass, copy(observations[np[1]]))
				# update the assignement
				new_z[np[1]] = size(partitions,1)+1
			end
		end
		push!(partitions, Partition(g, temp_ass))
	end

	# Re-assign observations
	for obs_idx in find(changed.==true)
		# New partition id
		new_p_idx = new_z[obs_idx]
		# Find old partition
		old_p_idx = old_z[obs_idx]
		# Get observation
		obs = copy(observations[obs_idx])
		# Remove obs from old partition
		filter!(x->x!=obs, partitions[ old_p_idx ].O )
		if new_p_idx < new_ass_idx
			# Add observation to new partition
			# For new partition, observation already moved earlier
			push!(partitions[new_p_idx].O, obs)
		end
	end

	# Find empty partitions and re-assign z
	to_remove = Array{Integer}(0)
	for (i,p) in reverse(collect(enumerate(partitions)))
		# Enumeration is reversed so that one can simply remove from high to low
		# without causing problems with indexes changing
		if size(p.O,1) == 0
			push!(to_remove, i)
			new_z[ new_z .>= i ] -= 1
		end
	end

	# if !isempty(to_remove)
	# 	@show to_remove[1]
	# 	@show size(partitions[1].O,1)
	# end
	# Remove partitions
	# _tally = tally(new_z)
	# to_remove = find(_tally .== 0)
	for (i,rm) in enumerate(to_remove)
		deleteat!(partitions, rm)
		# Was done before enumeration was reversed, keep for record
		# rm-(i-1) to account for the already removed partitions
		# deleteat!(partitions, rm-(i-1))
		println("Removed partition $(rm-(i-1))")
	end
end



# Ground truth
G = [goals_dict[62], goals_dict[68], goals_dict[8]]
z = vcat(fill(1,7), fill(2,6), fill(3,7))
partitions = []
push!(partitions, Partition(G[1], observations[1:7]))
push!(partitions, Partition(G[2], observations[8:13]))
push!(partitions, Partition(G[3], observations[14:20]))

calLoss(z, partitions, observations)
