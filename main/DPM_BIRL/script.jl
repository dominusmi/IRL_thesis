include("DPM_BIRL.jl")


srand(1)
mdp, policy = generate_gridworld(10,10,γ=0.9)
χ = generate_trajectories(mdp, policy, 50)
ϕ = eye(100)
learning_rate = 0.1
confidence = 1.0

c, EVD, log = DPM_BIRL(mdp, ϕ, χ, 30; α=learning_rate, β=confidence, ground_policy = policy, verbose = true, update = :ML)


using Plots
plotlyjs()
Plots.plot(EVD)

rewards = rewards_matrix(mdp)
heatmap(rewards')
heatmap(reshape(θ.values, (10,10)))


surface(rand(10), rand(10), rand(10))
