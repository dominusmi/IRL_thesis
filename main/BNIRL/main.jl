using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox
import StatsBase: sample
import Base: ==

include("../DPM_BIRL/DPM_BIRL.jl")
include("helper.jl")


### Initialise problem and generate trajectories
srand(1)
η, κ = 1.0, 1.0
mdp, policy = DPMBIRL.generate_diaggridworld(10,10,γ=0.999)
trajectories, z = DPMBIRL.generate_subgoals_trajectories(mdp, GridWorldState(2,1), [GridWorldState(6,2), GridWorldState(1,5)])
observations = traj2obs(mdp, trajectories)

# Setup general variables
n_states = size(states(mdp),1)-1
n_actions = size(actions(mdp),1)
n_observations = size(observations,1)
support_space = getSupportSpace(observations)
n_support_states = size(support_space,1)

### Precompute all Q-values and their πᵦ
tmp_dict, tmp_array = precomputeQ(mdp, support_space)

const state2goal = tmp_dict
const all_goals = tmp_array

logs = zeros(100,4)
srand(4)
goals = [sample(Goal) for i in 1:2]

dbg = [zeros(Integer, 10,10) for i in 1:2]

_log = []
max_iter = 1000
for t in 1:max_iter
	for (i,curr_goal) in enumerate(goals)
		assigned_to_goal = (z .== i)
		probs_vector = zeros(n_support_states)
		for obs in observations[assigned_to_goal]
			for (sᵢ, state) in enumerate(support_space)
				goal = state2goal[state]
				probs_vector[sᵢ] += likelihood(obs, goal, η)
			end
		end
		probs_vector /= sum(probs_vector)
		chosen = rand(Multinomial(1,probs_vector))
		state_chosen = support_space[findfirst(chosen)]
		goals[i] = state2goal[state_chosen]

		pos = DPMBIRL.i2s(mdp, state_chosen)

		dbg[1][11-pos[2], pos[1]] += (i==1 ? 1 : -1)
	end
	# @show get_state.(goals)
	push!(_log, partitioning_loss(goals, observations, z))
end
logs[:,4] = _log

Plots.plot(mean(logs,2))