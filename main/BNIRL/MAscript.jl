using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox
using Plots
include("../DPM_BIRL/DPM_BIRL.jl")
# reload("BNIRL")
include("BNIRL.jl")
using MABNIRL
pyplot()

### Initialise problem and generate trajectories
γ = 0.95
println("γ: $γ")
srand(5)
η, κ = 1.0, 0.1
mdp, policy = DPMBIRL.generate_diaggridworld(10,10,γ=γ)
# mdp.reward_states = mdp.reward_states[mdp.reward_values .> 0.]
# mdp.reward_values = [mdp.reward_values[i] > 0. ? 1.0 : 0.0 for i in 1:100]
# punishment = exp(inv(1-γ^2))
punishment = inv(1-γ^2)
trajectories, z = DPMBIRL.generate_subgoals_trajectories(mdp, GridWorldState(2,1), [GridWorldState(1,8), GridWorldState(10,10)])
observations = BNIRL.traj2obs(mdp, trajectories)

trajectories = [observations]

_log, glb = BNIRL.MABNIRL(mdp, trajectories, η, κ; max_iter=5_000, burn_in=2_000,
				use_assignements=true, ground_truth=z, punishment=punishment, use_clusters=true, n_goals=2)
