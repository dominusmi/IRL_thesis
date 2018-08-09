using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox
import StatsBase: sample
import Base: ==

include("../DPM_BIRL/DPM_BIRL.jl")


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

abstract type Prior end
immutable DiscretePrior <: Prior
	pdf::Array{<:AbstractFloat}
	prior::Distribution
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
	# if oᵢ.action == indmax(g.Q[oᵢ.state,:])
	# 	return 1
	# else
	# 	return 0
	# end
	denom = sum(exp.( η * g.Q[oᵢ.state, :] ))
	exp( η * g.Q[oᵢ.state, oᵢ.action] ) / denom
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

function sample(::Type{Goal}, p::Prior)
	global state2goal
	idx = findfirst(x->x!=0, rand(p.prior))
	# state2goal[idx]
	idx
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
tmp_dict = Dict()

# Solves mdp for each value
fig=nothing
for (i,state) in enumerate(find(x->x>0., support_space))
	raw_mdp = copy(mdp)
	victory_state_idx = state
	victory_state = GridWorldState(DPMBIRL.i2s(mdp,state)...)
	idx = find(x->x>0., raw_mdp.reward_values)
	raw_mdp.reward_values[idx] = 0.
	raw_mdp.reward_values[victory_state_idx] = 1.
	# raw_mdp.reward_values = [1.0]
	# raw_mdp.reward_states = [victory_state]
	raw_mdp.terminals = Set([victory_state])
	state_policy = DPMBIRL.solve_mdp(raw_mdp)
	if state == 62
		fig = heatmap(reshape(state_policy.util[1:end-1],(10,10))')
	end
	tmp_array[i] = state
	tmp_dict[state] = Goal(state, state_policy.qmat[1:end-1,:])
end
fig

const state2goal = tmp_dict
const index2state = tmp_array

# Find prior on support set
support_space_prior = support_space / sum(support_space)
goals_prior = DiscretePrior( support_space_prior, Multinomial(1, support_space_prior) )

# Initial assignement
z = vcat(fill(1,7), fill(2,6), fill(3,9))
srand(1)
curr_goals = [sample(Goal, goals_prior) for i in 1:3]

max_iter = 100
_logs = zeros(Integer, 10,10)
for t in 1:max_iter
	curr_goals = gibbs_sampling(curr_goals, z)
	dbg = map(x->DPMBIRL.i2s(mdp, x), curr_goals)
	# @show dbg[1]
	for pos in [dbg[3]]
		_logs[11-pos[2], pos[1]] += 1
	end
end

function gibbs_sampling(goals, z)
	global state2goal, index2state, n_support_states
	n_goals = size(curr_goals,1)
	# What happens in a gibbs sampler..
	for	goal_idx in 1:n_goals
		llh, priors = zeros(n_support_states), ones(n_support_states)
		assigned_to = (z .== goal_idx)
		for k in 1:n_support_states
			for obs in observations[assigned_to]
				llh[k] += likelihood(obs, state2goal[index2state[k]], 1.)
			end
			# priors[k] = prior( state2goal[index2state[k]], goals_prior )
		end
		# @show llh
		probs_vector = priors .* exp.(llh)
		# @show probs_vector / sum(probs_vector)
		chosen_idx = rand( Multinomial(1, probs_vector / sum(probs_vector)) )
		chosen_idx = findfirst(chosen_idx)
		goals[goal_idx] = index2state[chosen_idx]
	end
	goals
end


llhs = []
for obs in observations
	push!(llhs, likelihood(obs, state2goal[67], 1.0))
end
@show llhs
