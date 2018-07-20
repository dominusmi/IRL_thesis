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
