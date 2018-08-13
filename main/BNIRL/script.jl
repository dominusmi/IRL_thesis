using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox

include("../DPM_BIRL/DPM_BIRL.jl")
include("BNIRL.jl")
# reload("BNIRL")
using BNIRL

### Initialise problem and generate trajectories
srand(5)
η, κ = 1.0, 0.1
mdp, policy = DPMBIRL.generate_gridworld(10,10,γ=0.90)
# mdp.reward_states = mdp.reward_states[mdp.reward_values .> 0.]
# mdp.reward_values = [mdp.reward_values[i] > 0. ? 1.0 : 0.0 for i in 1:100]

trajectories, z = DPMBIRL.generate_subgoals_trajectories(mdp, GridWorldState(2,1), [GridWorldState(8,8), GridWorldState(1,4)])
observations = BNIRL.traj2obs(mdp, trajectories)

_log = BNIRL.main(mdp, observations, η, κ; max_iter=5000, burn_in=1000)

zs = zeros(Integer, 4000, 25)
for i in 1:4000
	zs[i,:] = _log[:z][i]
end
sizes = [ size(_log[:goals][i],1) for i in 1:4000 ]

histogram(sizes)
bar(sizes)

sum( [_log[:goals][sizes .== 2][i][1]==2 for i in 1:1000]) 
