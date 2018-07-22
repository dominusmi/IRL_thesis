# ∑ₜᵣₐⱼ ∑ₛₜₑₚ softmax(Q,a)
function trajectories_likelihood(mdp::GridWorld, π::Policy, trajectories::Array{MDPHistory}, η::Number; verbose = false)
    likelihood = 0.
    for (i,trajectory) in enumerate(trajectories)
        for state in trajectory.state_hist[1:end-2]

            # Get state index and optimal action (int)
            s = state_index(mdp, state)
            optimal = POMDPModels.a2int( action(π, state), mdp)+1

            # Compute optimal Q value for the step
            Qᵒ = π.qmat[s,optimal]
            nominator = exp(η*Qᵒ)

            denominator = 0.
            for a in POMDPModels.actions(mdp)
                # check if action inbounds
                if POMDPModels.inbounds(mdp, state, a)
                    a = POMDPModels.a2int( action(π, state), mdp )+1
                    Q = π.qmat[s,a]
                    denominator += exp(η*Q)
                end
            end
            likelihood -= log(nominator / denominator)
        end
    end
    likelihood
end

"""
    Trajectory likelihood given a policy
"""
function trajectory_likelihood(mdp::GridWorld, trajectory::MDPHistory, π::Policy; η::Number = 1.0)
    # Calculate likelihood trajectory
    likelihood = 0.0
    for state in trajectory.state_hist[1:end-2]

        # Get state index and optimal action (int)
        s = state_index(mdp, state)
        # optimal = POMDPModels.a2int( action(π, state), mdp)+1
        optimal = action_index(mdp, action(π,state))

        # Compute optimal Q value for the step
        Qᵒ = π.qmat[s,optimal]
        nominator = η*Qᵒ

        denominator = 1.0
        for a in POMDPModels.actions(mdp)
            # check if action inbounds
            if POMDPModels.inbounds(mdp, state, a)
                a = POMDPModels.a2int( action(π, state), mdp )+1
                Q = π.qmat[s,a]
                denominator += exp(η*Q)
            end
        end
        likelihood += nominator - log(denominator)
    end
    likelihood
end



"""
    Likelihood for single trajectory, recalculates optimal Q and modifies mdp and policy
"""
function trajectory_likelihood(real_mdp::GridWorld, trajectory::MDPHistory, reward::RewardFunction; η::Number = 1.0)
    # Set up new mdp with given reward functions
    mdp = copy(real_mdp)
    mdp.reward_values = zeros(mdp.size_x*mdp.size_y)
    for i in 1:mdp.size_x*mdp.size_y
        mdp.reward_values[i] = reward.values[i]
    end

    # Calculate optimal Q
    π = solve_mdp(mdp)

    # Calculate likelihood trajectory
    likelihood = 0.0
    for state in trajectory.state_hist[1:end-2]

        # Get state index and optimal action (int)
        s = state_index(mdp, state)
        optimal = POMDPModels.a2int( action(π, state), mdp)+1

        # Compute optimal Q value for the step
        Qᵒ = π.qmat[s,optimal]
        nominator = exp(η*Qᵒ)

        denominator = 1.0
        for a in POMDPModels.actions(mdp)
            # check if action inbounds
            if POMDPModels.inbounds(mdp, state, a)
                a = POMDPModels.a2int( action(π, state), mdp )+1
                Q = π.qmat[s,a]
                denominator += exp(η*Q)
            end
        end
        likelihood = -log(nominator / denominator)
    end
    likelihood
end
