"""
    Calculates inv(I-γPₚᵢ), where Pₚᵢ is the matrix of transition probabilities
    of size SxS
"""
function calInvTransition(mdp, πᵦ, γ)
    n_states = size(πᵦ,1)
    I = eye(n_states)
    Pₚᵢ = π2transition(mdp, πᵦ)
    inv( I-γ*Pₚᵢ )
end

"""
    Currently unused
"""
function caldQₖ!(dQₖ, mdp, invT, πᵦ, k, glb)
    states = ordered_states(mdp)
    πₐ = zeros(size(states,1)-1)

    # π "marginal"
    for s in states[1:end-1]
        si      = state_index(mdp, s)
        πₐ[si]  = sum( πᵦ[si,:] ) * glb.ϕ[si,k]
    end

    # dQ for each action
    for a in actions(mdp)
        ai = action_index(mdp,a)
        dQₖ[:,ai] = glb.ϕ[:,k] + mdp.discount_factor * glb.Pₐ[ai] * invT * πₐ
    end
end


"""
    Calculates the LOG likelihood of a cluster
"""
function cal𝓛(mdp, πᵦ, χ, glb)
    𝓛  = 0.
    for (m,trajectory) in enumerate(χ)
        traj_size = size(trajectory.state_hist,1)-1
        𝓛 += trajectory_likelihood(mdp, trajectory, πᵦ, glb)
        # for (h,state) in enumerate(trajectory.state_hist[1:end-1])
        #     sₕ = state_index(mdp, state)
        #     aₕ = action_index(mdp, trajectory.action_hist[h])
        #     𝓛 += log(πᵦ[sₕ,aₕ])
        # end
    end
    𝓛
end

"""
   Returns the likelihood and the gradient of the likelihood given a Boltzmann
   policy and a set of trajectories
"""
function cal∇𝓛(mdp, invT, πᵦ, χ, glb::Globals)
    # 𝓛  = 0.
    ∇𝓛 = zeros(glb.n_features)
    for k in 1:glb.n_features
        dQₖ = zeros( glb.n_states, glb.n_actions )
        caldQₖ!(dQₖ, mdp, invT, πᵦ, k, glb)

        # Calculates total gradient over trajectories
        for (m,trajectory) in enumerate(χ)
            traj_size = size(trajectory.state_hist,1)-1
            for (h,state) in enumerate(trajectory.state_hist[1:end-1])
                sₕ = state_index(mdp, state)
                aₕ = action_index(mdp, trajectory.action_hist[h])

                # 𝓛 += state_action_lh(πᵦ,sₕ,aₕ)
                # 𝓛 += state_action_lh(πᵦ,sₕ,aₕ) / traj_size

                dl_dθₖ = glb.β * ( dQₖ[sₕ,aₕ] - sum( [ state_action_lh(πᵦ,sₕ,ai⁻) * dQₖ[sₕ,ai⁻] for ai⁻ in glb.actions_i ] ) )
                ∇𝓛[k] += dl_dθₖ
            end
        end
    end
    ∇𝓛
end
