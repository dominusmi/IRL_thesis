using Plots
using LaTeXStrings
using POMDPs
using POMDPModels
using Distributions
using POMDPToolbox
using DiscreteValueIteration
import Base.copy

manhattan(x,y) = sum(abs.(x-y))


function generate_reward_states(x,y)
    states = []

    # Generate terminal
    tx, ty = rand(1:x), rand(1:y)
    push!(states, (tx,ty,1.0))

    # Generate negative
    n_bad = Int(round(0.25*x*y))
    for i in 1:n_bad
        while true
            bx,by = rand(1:x), rand(1:y)
            if (bx,by) !== (tx,ty)
                push!(states, (bx,by,-0.5))
                break
            end
        end
    end
    states
end

function solve_mdp(mdp::GridWorld)
    solver = ValueIterationSolver(max_iterations=100, belres=1e-3)
    policy = ValueIterationPolicy(mdp)
    policy = solve(solver, mdp, policy)
end

function generate_gridworld(size_x::Integer, size_y::Integer;
                            γ=0.9, boundary_penalty=-1.0, transitionₚ=1.0)

    reward_states = generate_reward_states(size_x, size_y)

    mdp = GridWorld(size_x,                                                                 # size x
                    size_y,                                                                 # size y
                    map(x->GridWorldState(x[1:2]...), reward_states),                       # Reward states
                    map(x->x[3], reward_states),                                            # Respective rewards
                    boundary_penalty,                                                       # Boundary penalty
                    transitionₚ,                                                            # Transition probability
                    Set([GridWorldState(x[1:2]...) for x in reward_states if x[3]>0]),      # Terminal states
                    γ                                                                       # Discount factor γ
                )
    mdp, solve_mdp(mdp)
end

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
            manhattan([x,y], [3,3]) > 1 && break
        end

        # Generate trajectories
        traj = sim(mdp, GridWorldState(x,y), max_steps=15) do s
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
    Solves mdp with new reward function
"""
function solve_mdp(mdp::GridWorld, r::RewardFunction)
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
