include("../reward.jl")
include("../utilities/gridworld.jl")
include("../utilities/policy.jl")
include("../utilities/general.jl")


"""
    (proportional) Likelihood function for a single state action
    Normally should have normalisiation, but not important when calculating ∇𝓛
"""
state_action_lh(πᵦ, s,a) = πᵦ[s,a]

"""
    Executes a maximum likelihood inverse reinforcement learning
    mdp:            the problem
    ϕ:              operator to features space
    trajectories:   experts' trajectories
    iterations:     number of iterations to run for
    α:              learning rate
    β:              confidence parameter
"""
function MLIRL(mdp, ϕ, trajectories, iterations; α=0.1, β=0.5, ground_policy = nothing, verbose=true)

    θ = sample(RewardFunction, size(ϕ,2))
    γ = mdp.discount_factor
    states = ordered_states(mdp)

    n_states  = size(states,1)-1
    n_actions = size( actions(mdp),1 )
    n_features = size(ϕ,2)

    EVD = []

    if verbose && ground_policy !== nothing
        v = policy_evaluation(mdp, ground_policy)
    end

    for t in 1:iterations
        tic()
        # π, Qᵣ   = calQᵣ(mdp, θ)
        π  = solve_mdp(mdp, θ)
        πᵦ = calπᵦ(mdp, π.qmat, β)

        # Log EVD
        if verbose && ground_policy !== nothing
            tic()
            # Need to change this to account for features
            πᵣ = solve_mdp(mdp, θ)
            vᵣ = policy_evaluation(mdp, πᵣ)
            push!(EVD, norm(v-vᵣ))
            elapsed = toq()
        end


        invT = calInvTransition(mdp, πᵦ, γ)
        actions_i = action_index.(mdp, actions(mdp))
        dQ = zeros(n_states, n_actions)
        ∇𝓛 = SharedArray{Float64,1}(zeros(n_features))

        # Precpmputes transition matrix for all actions
        # (independent of features)
        Pₐ = a2transition.(mdp,actions(mdp))


        # Calculates gradient per feature
        for k in 1:n_features
            dQₖ = zeros( n_states, n_actions )
            caldQₖ!(dQₖ, mdp, ϕ, invT, Pₐ, πᵦ, k)
            # Calculates total gradient over trajectories
            for (m,trajectory) in enumerate(trajectories)
                for (h,state) in enumerate(trajectory.state_hist[1:end-1])
                    sₕ = state_index(mdp, state)
                    aₕ = action_index(mdp, trajectory.action_hist[h])

                    dl_dθₖ = β * ( dQₖ[sₕ,aₕ] - sum( [ state_action_lh(πᵦ,sₕ,ai⁻) * dQₖ[sₕ,ai⁻] for ai⁻ in actions_i ] ) )
                    ∇𝓛[k] += dl_dθₖ
                end
            end
        end
        θ += α * ∇𝓛
        # θ.values /= maximum(θ.values)
        elapsed = toq()

        verbose ? println("Iteration took $elapsed seconds") : nothing
    end

    # Log EVD
    if verbose && ground_policy !== nothing
        # Need to change this to account for features
        πᵣ = solve_mdp(mdp, θ)
        vᵣ = policy_evaluation(mdp, πᵣ)
        push!(EVD, norm(v-vᵣ))
        println("Final EVD: $(EVD[end])")
    end

    θ, EVD
end
