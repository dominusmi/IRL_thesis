#################################################################
# This file implements an alternative of the normal GridWorld
# where the agent is also allowed to take diagonal actions
#################################################################

#################################################################
# States and Actions
#################################################################
# state of the agent in grid world
struct GridWorldState # this is not immutable because of how it is used in transition(), but maybe it should be
	x::Int64 # x position
	y::Int64 # y position
    done::Bool # entered the terminal reward state in previous step - there is only one terminal state
    GridWorldState(x,y,done) = new(x,y,done)
    GridWorldState() = new()
end

# simpler constructors
GridWorldState(x::Int64, y::Int64) = GridWorldState(x,y,false)
# for state comparison
function ==(s1::GridWorldState,s2::GridWorldState)
    if s1.done && s2.done
        return true
    elseif s1.done || s2.done
        return false
    else
        return posequal(s1, s2)
    end
end
# for hashing states in dictionaries in Monte Carlo Tree Search
posequal(s1::GridWorldState, s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y
function hash(s::GridWorldState, h::UInt64 = zero(UInt64))
    if s.done
        return hash(s.done, h)
    else
        return hash(s.x, hash(s.y, h))
    end
end
Base.copy!(dest::GridWorldState, src::GridWorldState) = (dest.x=src.x; dest.y=src.y; dest.done=src.done; return dest)

# action taken by the agent indicates desired travel direction
const GridWorldAction = Symbol # deprecated - this is here so that other people's code won't break

#################################################################
# Grid World MDP
#################################################################
# the grid world mdp type
mutable struct DiagGridWorld <: MDP{GridWorldState, Symbol}
	size_x::Int64 # x size of the grid
	size_y::Int64 # y size of the grid
	reward_states::Vector{GridWorldState} # the states in which agent recieves reward
	reward_values::Vector{<:Number} # reward values for those states
    bounds_penalty::Float64 # penalty for bumping the wall (will be added to reward)
    tprob::Float64 # probability of transitioning to the desired state
    terminals::Set{GridWorldState}
    discount_factor::Float64 # disocunt factor
end
# we use key worded arguments so we can change any of the values we pass in
function DiagGridWorld(sx::Int64, # size_x
                   sy::Int64; # size_y
                   rs::Vector{GridWorldState}=[GridWorldState(4,3), GridWorldState(4,6), GridWorldState(9,3), GridWorldState(8,8)],
                   rv::Vector{<:Number}=[-10.,-5,10,3],
                   penalty::Float64=0.0, # penalty for trying to go out of bounds  (will be added to reward)
                   tp::Float64=0.7, # tprob
                   discount_factor::Float64=0.95,
                   terminals=Set{GridWorldState}([rs[i] for i in filter(i->rv[i]>0.0, 1:length(rs))]))
    return DiagGridWorld(sx, sy, rs, rv, penalty, tp, Set{GridWorldState}(terminals), discount_factor)
end

DiagGridWorld(;sx::Int64=10, sy::Int64=10, kwargs...) = GridWorld(sx, sy; kwargs...)

# convenience function
function term_from_rs(rs, rv)
    terminals = Set{GridWorldState}()
    for (i,v) in enumerate(rv)
        if v > 0.0
            push!(terminals, rs[i])
        end
    end
end


#################################################################
# State and Action Spaces
#################################################################
# This could probably be implemented more efficiently without vectors

function states(mdp::DiagGridWorld)
    s = vec(collect(GridWorldState(x, y, false) for x in 1:mdp.size_x, y in 1:mdp.size_y))
    push!(s, GridWorldState(0, 0, true))
    return s
end

actions(mdp::DiagGridWorld) = [:n, :nw, :w, :sw, :s, :se, :e, :ne]

n_states(mdp::DiagGridWorld) = mdp.size_x*mdp.size_y+1
n_actions(mdp::DiagGridWorld) = 4

function reward(mdp::DiagGridWorld, state::GridWorldState, action::Symbol)
    if state.done
        return 0.0
    end
    r = static_reward(mdp, state)
    if !inbounds(mdp, state, action)
        r += mdp.bounds_penalty
    end
	return r
end

"""
    static_reward(mdp::GridWorld, state::GridWorldState)

Return the reward for being in the state (the reward not including bumping)
"""
function static_reward(mdp::DiagGridWorld, state::GridWorldState)
	r = 0.0
	n = length(mdp.reward_states)
	for i = 1:n
		if posequal(state, mdp.reward_states[i])
			r += mdp.reward_values[i]
		end
	end
    return r
end

#checking boundries- x,y --> points of current state
inbounds(mdp::DiagGridWorld,x::Int64,y::Int64) = 1 <= x <= mdp.size_x && 1 <= y <= mdp.size_y
inbounds(mdp::DiagGridWorld,state::GridWorldState) = inbounds(mdp, state.x, state.y)

"""
    inbounds(mdp::GridWorld, s::GridWorldState, a::Symbol)

Return false if `a` is trying to go out of bounds, true otherwise.
"""
function inbounds(mdp::DiagGridWorld, s::GridWorldState, a::Symbol)
    xdir = s.x
    ydir = s.y
    if a == :n
        ydir += 1
    elseif a == :nw
        ydir += 1
        xdir -= 1
    elseif a == :w
        xdir -= 1
    elseif a == :sw
        xdir -= 1
        ydir -= 1
    elseif a == :s
        ydir -= 1
    elseif a == :se
        xdir += 1
        ydir -= 1
    elseif a == :e
        xdir += 1
    elseif a == :ne
        xdir += 1
        ydir += 1
    end
    return inbounds(mdp, GridWorldState(xdir, ydir, s.done))
end

function fill_probability!(p::AbstractVector{Float64}, val::Float64, index::Int64)
	for i = 1:length(p)
		if i == index
			p[i] = val
		else
			p[i] = 0.0
		end
	end
end

function transition(mdp::DiagGridWorld, state::GridWorldState, action::Symbol)

	a = action
	x = state.x
	y = state.y

    neighbors = MVector(
        GridWorldState(x, y+1, false), # north
        GridWorldState(x+1, y+1, false), # north east
        GridWorldState(x+1, y, false), # east
        GridWorldState(x+1, y-1, false), # south east
        GridWorldState(x, y-1, false), # south
        GridWorldState(x-1, y-1, false), # south west
        GridWorldState(x-1, y, false), # south west
        GridWorldState(x-1, y+1, false), # north west
        GridWorldState(x, y, false)    # stay
       )

    probability = MVector{9, Float64}()
    fill!(probability, 0.0)

    if state.done
        fill_probability!(probability, 1.0, 9)
        neighbors[9] = GridWorldState(x, y, true)
        return SparseCat(neighbors, probability)
    end

    reward_states = mdp.reward_states
    reward_values = mdp.reward_values
	n = length(reward_states)
    if state in mdp.terminals
		fill_probability!(probability, 1.0, 9)
        neighbors[9] = GridWorldState(x, y, true)
        return SparseCat(neighbors, probability)
    end

    # The following match the definition of neighbors
    # given above
    target_neighbor = 0
    if a == :n
        target_neighbor = 1
	elseif a == :ne
        target_neighbor = 2
	elseif a == :e
        target_neighbor = 3
	elseif a == :se
        target_neighbor = 4
    elseif a == :s
        target_neighbor = 5
    elseif a == :sw
        target_neighbor = 6
    elseif a == :w
        target_neighbor = 7
    elseif a == :nw
        target_neighbor = 8
	end
    # @assert target_neighbor > 0

	if !inbounds(mdp, neighbors[target_neighbor])
        # If would transition out of bounds, stay in
        # same cell with probability 1
		fill_probability!(probability, 1.0, 9)
	else
		probability[target_neighbor] = mdp.tprob

        oob_count = 0 # number of out of bounds neighbors

        for i = 1:length(neighbors)
             if !inbounds(mdp, neighbors[i])
                oob_count += 1
                @assert probability[i] == 0.0
             end
        end

        new_probability = (1.0 - mdp.tprob)/(4-oob_count)

        for i = 1:8 # do not include neighbor 9
            if inbounds(mdp, neighbors[i]) && i != target_neighbor
                probability[i] = new_probability
            end
        end
	end

    return SparseCat(neighbors, probability)
end


function action_index(mdp::DiagGridWorld, a::Symbol)
    # lazy, replace with switches when they arrive
    if a == :n
        return 1
    elseif a == :nw
        return 2
    elseif a == :w
        return 3
    elseif a == :sw
        return 4
    elseif a == :s
        return 5
    elseif a == :se
        return 6
    elseif a == :e
        return 7
    elseif a == :ne
        return 8
    else
        error("Invalid action symbol $a")
    end
end


function state_index(mdp::DiagGridWorld, s::GridWorldState)
    return s2i(mdp, s)
end

function s2i(mdp::DiagGridWorld, state::GridWorldState)
    if state.done
        return mdp.size_x*mdp.size_y + 1
    else
        return sub2ind((mdp.size_x, mdp.size_y), state.x, state.y)
    end
end

#=
function i2s(mdp::GridWorld, i::Int)
end
=#

isterminal(mdp::DiagGridWorld, s::GridWorldState) = s.done

discount(mdp::DiagGridWorld) = mdp.discount_factor

convert_s(::Type{A}, s::GridWorldState, mdp::DiagGridWorld) where A<:AbstractArray = Float64[s.x, s.y, s.done]
convert_s(::Type{GridWorldState}, s::AbstractArray, mdp::DiagGridWorld) = GridWorldState(s[1], s[2], s[3])

function a2int(a::Symbol, mdp::DiagGridWorld)
    action_index(mdp, a)
end

function int2a(a::Int, mdp::DiagGridWorld)
    if a == 0
        return :n
    elseif a == 1
        return :nw
    elseif a == 1
        return :w
    elseif a == 1
        return :sw
    elseif a == 1
        return :s
    elseif a == 1
        return :se
    elseif a == 1
        return :e
    elseif a == 1
        return :ne
    else
        throw("Action $a is invalid")
    end
end

convert_a(::Type{A}, a::Symbol, mdp::DiagGridWorld) where A<:AbstractArray = [Float64(a2int(a, mdp))]
convert_a(::Type{Symbol}, a::A, mdp::DiagGridWorld) where A<:AbstractArray = int2a(Int(a[1]), mdp)

initial_state(mdp::DiagGridWorld, rng::AbstractRNG) = GridWorldState(rand(rng, 1:mdp.size_x), rand(rng, 1:mdp.size_y))
