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
    Non-logged likelihood of a single state-action pair
"""
function state_action_likelihood(mdp::GridWorld, state_action::Tuple, πᵦ::Array{Float64,2}, glb)
    # Calculate likelihood trajectory
    sₕ = state_index(mdp, state_action[1])
    aₕ = action_index(mdp, state_action[2])
    πᵦ[sₕ,aₕ]
end


"""
    Trajectory likelihood given a policy
"""
function trajectory_likelihood(mdp::GridWorld, trajectory::MDPHistory, πᵦ::Array{Float64,2}, glb)
    # Calculate likelihood trajectory
    log_likelihood = 0.
    traj_size = size(trajectory.state_hist,1)-1
    for (h,state) in enumerate(trajectory.state_hist[1:end-1])
        sₕ = state_index(mdp, state)
        aₕ = action_index(mdp, trajectory.action_hist[h])
        log_likelihood += log(πᵦ[sₕ,aₕ])
    end
    log_likelihood /glb.n_trajectories
end



"""
    Likelihood for single trajectory, recalculates optimal Q and modifies mdp and policy
"""
function trajectory_likelihood(real_mdp::GridWorld, trajectory::MDPHistory, reward::RewardFunction, glb::Globals)
    # Set up new mdp with given reward functions
    mdp = copy(real_mdp)
    mdp.reward_values = zeros(mdp.size_x*mdp.size_y)
    for i in 1:mdp.size_x*mdp.size_y
        mdp.reward_values[i] = reward.values[i]
    end

    # Calculate optimal Q
    π = solve_mdp(mdp)
    πᵦ = calπᵦ(mdp,π.qmat,glb)

    log_likelihood = trajectory_likelihood(mdp, trajectory, πᵦ, glb)
    log_likelihood
end
