using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox
import StatsBase: sample
import Base: ==

include("../DPM_BIRL/DPM_BIRL.jl")
include("helper.jl")


### Initialise problem and generate trajectories
srand(5)
η, κ = 1.0, 1.0
mdp, policy = DPMBIRL.generate_gridworld(10,10,γ=0.90)
# mdp.reward_states = mdp.reward_states[mdp.reward_values .> 0.]
# mdp.reward_values = [mdp.reward_values[i] > 0. ? 1.0 : 0.0 for i in 1:100]

trajectories, z = DPMBIRL.generate_subgoals_trajectories(mdp, GridWorldState(2,1), [GridWorldState(7,2), GridWorldState(8,8), GridWorldState(1,4)])
observations = traj2obs(mdp, trajectories)

# heatmap(reshape(mdp.reward_values,(10,10)))

# Setup general variables
n_states = size(states(mdp),1)-1
n_actions = size(actions(mdp),1)
n_observations = size(observations,1)
support_space = getSupportSpace(observations)
n_support_states = size(support_space,1)

### Precompute all Q-values and their πᵦ
tmp_dict, tmp_array, utils = precomputeQ(mdp, support_space)

const state2goal = tmp_dict
const all_goals = tmp_array

logs = zeros(100,4)
srand(4)
goals = [sample(Goal) for i in 1:3]

dbg = [zeros(Integer, 10,10) for i in 1:2]

_log = []
max_iter = 5000
goal_hist = zeros(Integer, max_iter, 3)
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

		# dbg[1][11-pos[2], pos[1]] += (i==1 ? 1 : -1)
	end

	goal_hist[t,1] = goals[1].state
	goal_hist[t,2] = goals[2].state
	goal_hist[t,3] = goals[3].state
	# @show get_state.(goals)
	push!(_log, partitioning_loss(goals, observations, z))
end


df = DataFrame()
df[:states] = collect(keys(t))

""" Get a frequency dictionary """
t = Dict([(i,count(x->x==i,goal_hist[:,3])/max_iter) for i in support_space])
df[:p_g3] = collect(values(t))
sort!(df, cols=[:states])



Plots.gr()
labels = string.(collect(keys(t)))
bar(labels, collect(values(t)), rotation=45, xtickfont = font(5, "Courier"))


for key in keys(t)
	println("$(t[key])")
end


CSV.write("dataframe_3_goals.csv", df)
