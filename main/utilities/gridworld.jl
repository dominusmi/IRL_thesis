using DiscreteValueIteration
import Base.copy

manhattan(x,y) = sum(abs.(x-y))

function generate_reward_states(x,y)

    states = zeros(y,x)
    terminals = []

    # Generate terminal
    tx, ty = rand(1:x), rand(1:y)
    push!(terminals, (tx,ty,1.0))
    states[ty, tx] = 1.0

    # Generate negative
    n_bad = Int(round(0.25*x*y))
    for i in 1:n_bad
        while true
            bx,by = rand(1:x), rand(1:y)
            if (bx,by) !== (tx,ty)
                states[by, bx] = -0.5
                break
            end
        end
    end
    states, terminals
end

function solve_mdp(mdp::GridWorld)
    solver = ValueIterationSolver(max_iterations=100, belres=1e-3)
    policy = ValueIterationPolicy(mdp)
    policy = solve(solver, mdp, policy)
end


function generate_gridworld(size_x::Integer, size_y::Integer;
                            γ=0.9, boundary_penalty=-1.0, transitionₚ=1.0)

    states, terminals = generate_reward_states(size_x, size_y)

    all_states = reshape([GridWorldState(x,y) for x in 1:size_x, y in 1:size_y], size_x*size_y)
    rewards = reshape([states[y,x] for x in 1:size_x, y in 1:size_y], size_x*size_y)

    mdp = GridWorld(size_x,                                                                 # size x
                    size_y,                                                                 # size y
                    # map(x->GridWorldState(x[1:2]...), reward_states),                       # Reward states
                    # map(x->x[3], reward_states),                                            # Respective rewards
                    all_states,
                    rewards,
                    boundary_penalty,                                                       # Boundary penalty
                    transitionₚ,                                                            # Transition probability
                    Set([GridWorldState(x[1:2]...) for x in terminals]),      # Terminal states
                    γ                                                                       # Discount factor γ
                )
    mdp, solve_mdp(mdp)
end

# function generate_gridworld(size_x::Integer, size_y::Integer;
#                             γ=0.9, boundary_penalty=-1.0, transitionₚ=1.0)
#
#     reward_states = generate_reward_states(size_x, size_y)
#
#     mdp = GridWorld(size_x,                                                                 # size x
#                     size_y,                                                                 # size y
#                     map(x->GridWorldState(x[1:2]...), reward_states),                       # Reward states
#                     map(x->x[3], reward_states),                                            # Respective rewards
#                     boundary_penalty,                                                       # Boundary penalty
#                     transitionₚ,                                                            # Transition probability
#                     Set([GridWorldState(x[1:2]...) for x in reward_states if x[3]>0]),      # Terminal states
#                     γ                                                                       # Discount factor γ
#                 )
#     mdp, solve_mdp(mdp)
# end

function copy(mdp::GridWorld)
    new_mdp = GridWorld(mdp.size_x,
                        mdp.size_y,
                        copy(mdp.reward_states),
                        copy(mdp.reward_values),
                        mdp.bounds_penalty,
                        mdp.tprob,
                        copy(mdp.terminals),
                        mdp.discount_factor)
end

"""
    Generate trajectories given an mdp and policy
"""
function generate_trajectories(mdp, policy, n=50)
    x,y = 0,0
    trajectories = Vector{MDPHistory}(0)
    for i = 1:n
        # Start at least 2 blocks away
        while true
            x = rand(1:mdp.size_x)
            y = rand(1:mdp.size_y)
            manhattan([x,y], [3,3]) > 3 && break
        end

        # Generate trajectories
        traj = sim(mdp, GridWorldState(x,y), max_steps=1000) do s
            a = action(policy, s)

            a_index = POMDPModels.a2int(a, mdp)+1
            s_index = state_index(mdp, s)

            if indmax(policy.qmat[s_index,:]) != a_index
                error("Policy does not maximise Q")
            end

            return a
        end
        push!(trajectories, traj)
    end
    trajectories
end

"""
    Returns a new mdp with the reward function's values
"""
function reward_mdp(mdp::GridWorld, r::RewardFunction)
    new_mdp = copy(mdp)
    n_states = size(r.values,1)
    reward_states = Array{GridWorldState}(n_states)
    reward_values = zeros(n_states)

    sizes = (mdp.size_x, mdp.size_y)

    for s in 1:n_states
        pos = ind2sub(sizes, s)
        reward_states[s] = GridWorldState(pos...)
        reward_values[s] = r.values[s]
    end
    new_mdp.reward_states = reward_states
    new_mdp.reward_values = reward_values
    new_mdp
end


"""
    Solves mdp with new reward function
"""
function solve_mdp(mdp::GridWorld, r::RewardFunction)
    new_mdp = reward_mdp(mdp, r)
    new_π = solve_mdp(new_mdp)
    new_π
end


"""
    Given a state index (between 1 and SxS), returns the state neighnours in index
    notation
"""
function state_neighbours(mdp,s)
    as = [[0,1],[0,-1],[-1,0],[1,0]]

    x,y = ind2sub( (mdp.size_x, mdp.size_y), s )

    state_indeces = zeros(Integer,4)
    for (i,a) in enumerate(as)
        state_indeces[i] = sub2ind((mdp.size_x, mdp.size_y), x+a[1], y+a[2])
        state_indeces[i] = state_indeces[i] > 0 ? state_indeces[i] : s
        state_indeces[i] = state_indeces[i] < mdp.size_x* mdp.size_y ? state_indeces[i] : s
    end
    state_indeces
end


"""
    Index to state coordinates
"""
i2s(mdp, index) = ind2sub((mdp.size_x, mdp.size_y), index)

"""
    Tranforms grid world reward states into matrix
"""
function rewards_matrix(mdp::GridWorld)
    reward_matrix = zeros(mdp.size_x,mdp.size_y)
    for i in 1:size(mdp.reward_states,1)
        x = mdp.reward_states[i].x
        y = mdp.reward_states[i].y

        r = mdp.reward_values[i]

        reward_matrix[y,x] = r
    end
    reward_matrix
end


"""
    Given an action, generates the SxS probability transition matrix Pₐ
"""
function a2transition(mdp, a)
    states = ordered_states(mdp)
    n_states = size(states,1)-1
    Pₐ = zeros(n_states, n_states)

    for s in states[1:end-1]
        si = state_index(mdp,s)
        states⁻ = transition(mdp, s, a)
        states⁻, p = states⁻.vals, states⁻.probs
        for (j,s⁻) in enumerate(states⁻)
            if isterminal(mdp, s⁻)
                continue
            end
            s⁻ = POMDPModels.inbounds(mdp, s⁻) ? s⁻ : s
            si⁻ = state_index(mdp, s⁻)
            Pₐ[si, si⁻] = p[j]
        end
    end
    Pₐ
end


"""
    Calculates the log likelihood given a Q-value
"""
function log_likelihood(mdp::GridWorld, Q::Array{<:AbstractFloat,2}, trajectories::Array{<:MDPHistory})
    llh = 0.
    BoltzmannQ = Q .- log.(sum(exp.(Q),2))

    for (i,trajectory) in enumerate(trajectories)
        normalising = size(trajectory.state_hist,1)-1
        for (i,state) in enumerate(trajectory.state_hist[1:end-1])
            s = state_index(mdp, state)
            a = action_index(mdp, trajectory.action_hist[i])
            llh += BoltzmannQ[s,a] / normalising
        end
    end
    llh
end

function generate_path_trajectories(mdp, states)
    state_counter = 2
    mdp.reward_values[ state_index(mdp,states[state_counter])] = 1.0

    policy = solve_mdp(mdp)
    traj = sim(mdp, states[1], max_steps=1000) do s
        if s ∈ states && s != states[end] && s != states[1]
            state_counter += 1
            mdp.reward_values[ mdp.reward_values .> 0. ] = 0.
            mdp.reward_values[ state_index(mdp,states[state_counter]) ] = 1.0
            policy = solve_mdp(mdp)
            # println("$s, $state_counter")
        end
        a = action(policy, s)

        a_index = POMDPModels.a2int(a, mdp)+1
        s_index = state_index(mdp, s)

        return a
    end
end


function generate_subgoals_trajectories(mdp)

    # TODO: this should remove the positive reward but keep the negative
    tmp_mdp = copy(mdp)
    tmp_mdp.reward_values[ tmp_mdp.reward_values .> 0. ] = 0.

    states = [GridWorldState(1,1), GridWorldState(2,7), GridWorldState(8,7), GridWorldState(8,1)]

    tmp_mdp.terminals =  Set([states[end]])
    generate_path_trajectories(tmp_mdp, states)
end


# function gridworld_heatmap(mdp)
#     Plots.heatmap(reshape(mdp.reward_values,(10,10)))
# end
