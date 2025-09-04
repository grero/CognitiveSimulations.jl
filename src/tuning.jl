function estimate_place_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T}, y::AbstractArray{T,3}) where T <: Real
    ncells,nb,nt = size(h)
    zp = Vector{Dict{Tuple{T,T},T}}(undef, ncells)
    for i in 1:ncells
        zp[i] = estimate_place_tuning(trialstruct, y[i,:,:], y)
    end
    zp
end

function estimate_place_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, y::AbstractArray{T,3},idxe::AbstractVector{Int64}) where T <: Real
     nb,nt = size(h)
    zp = Dict{Tuple{T,T},T}()
    for i in 1:nt
        for j in 1:idxe[i]
            _x = tuple(y[:,j,i]...)
            zp[_x] = get(zp, _x, zero(T)) + h[j,i]
        end
    end

    vv = collect(values(zp))
    vs = sum(vv)
    for (k,v) in zp
        zp[k] = v/vs
    end
    zp
end

function estimate_view_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, idxe::AbstractVector{Int64};nbins::Union{Int64, Nothing}=nothing) where T <: Real
    nb,nt = size(h)    
    z = zeros(T, size(x,1))
    zp = zeros(T, size(x,1))
    hq = zero(T)
    xq = zero(T)
    θ = trialstruct.angular_pref.μ
    for i in 1:nt
        for j in 1:idxe[i]
            z .+= h[j,i]*x[:,j,i]
            hq += h[j,i]
            zp .+= x[:,j,i]
            xq += sum(x[:,j,i])
        end
    end
    z ./= sum(z)
    zp ./= sum(zp)
    zp, z, θ
end

function estimate_view_by_place_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64};nbins::Union{Int64, Nothing}=nothing) where T <: Real
    # estimate overall place tuning first to get the locations
    # this can be done faster
    nt = size(h,2)
    zp = estimate_place_tuning(trialstruct, h, y, idxe)
    vq = Dict{Tuple{T,T},Tuple{Vector{T}, Vector{T}}}()
    θ = trialstruct.angular_pref.μ
    for k in keys(zp)
        zv = zeros(T, length(θ))
        for i in 1:nt
            bidx = findall(dropdims(mapslices(_y->all(_y.==k), y[:,1:idxe[i],i], dims=1),dims=1))
            for j in bidx
                zv .+= h[j,i].*x[:,j,i]
            end
        end
        #zv ./= sum(zv)
        vq[k] = (θ, zv)
    end
    vq
end

function estimate_view_by_place_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::Dict{Tuple{T,T},Vector{CartesianIndex{2}}}) where T <: Real
    nbins = Dict{Tuple{T,T},Int64}()
    for (k,v) in idxe
        nbins[k] = length(v)
    end
    estimate_view_by_place_tuning(trialstruct, h, x, y, idxe, nbins)
end

function estimate_view_by_place_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::Dict{Tuple{T,T},Vector{CartesianIndex{2}}},nbins::Dict{Tuple{T,T}, Int64}) where T <: Real
    # estimate overall place tuning first to get the locations
    # this can be done faster
    nt = size(h,2)
    vq = Dict{Tuple{T,T},Tuple{Vector{T}, Vector{T}}}()
    θ = trialstruct.angular_pref.μ
    for (k,v) in idxe
        zv = zeros(T, length(θ))
        zp = zeros(T, length(θ))
        nn = get(nbins, k, length(v))
        if nn < length(v)
            vv = shuffle(v)[1:nn]
        else
            vv = v
        end
        for j in vv 
            zv .+= h[j].*x[:,j]
            zp .+= x[:,j]
        end
        #normalize by the occupancy
        zv ./= zp
        vq[k] = (θ, zv)
    end
    vq
end

function get_view_by_place_bins(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    idx = Dict{Tuple{T,T},Vector{CartesianIndex{2}}}()
    idxs = Dict{Tuple{T,T},Vector{CartesianIndex{2}}}()
    zp = estimate_place_tuning(trialstruct, h, y, idxe)
    nt = size(x,3)
    for k in keys(zp)
        idx[k] = Int64[]
        idxs[k] = Int64[] 
        for i in 1:nt
            bidx = findall(dropdims(mapslices(_y->all(_y.==k), y[:,1:idxe[i],i], dims=1),dims=1))
            append!(idx[k],[CartesianIndex((b,i)) for b in bidx])
            bidxl = findall(dropdims(mapslices(_y->all(_y.!=k), y[:,1:idxe[i],i], dims=1),dims=1))
            append!(idxs[k],[CartesianIndex((b,i)) for b in bidxl])
        end
    end
    zp, idx, idxs
end



function estimate_view_by_place_tuning_interaction(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64};nruns=1000) where T <: Real
    # get view tuning for each place
    zp,idx,idxs = CognitiveSimulations.get_view_by_place_bins(trialstruct, h, x, y, idxe)

    vq = estimate_view_by_place_tuning(trialstruct, h, x, y, idx)
    h1,h12 = estimate_view_by_place_tuning_interaction(vq)
    nbins = Dict(k=>length(v) for (k,v) in idx)
    # shuffled
    h1s = zeros(T, nruns)
    h12s = zeros(T, nruns)
    for i in 1:nruns
        vqs = estimate_view_by_place_tuning(trialstruct, h, x, y, idxs, nbins)
        h1s[i], h12s[i] = estimate_view_by_place_tuning_interaction(vqs)
    end
    h1-h12, h1s.-h12s
end

function estimate_view_by_place_tuning_interaction(vq::Dict{Tuple{T,T}, Tuple{Vector{T}, Vector{T}}}) where T <: Real
    locations = collect(keys(vq))
    pl = zeros(T,length(locations))
    h12 = zero(T)
    vz = zeros(T, length(vq[locations[1]][1]))
    for (i,l) in enumerate(locations)
        _,_vz = vq[l]
        _pl = sum(_vz)
        # skip locations with no activity
        if _pl == zero(T)
            continue
        end
        pl[i] = _pl
        _vz ./= sum(_vz) #normalize
        h12 += sum(filter(isfinite,_vz.*log2.(_vz).*pl[i]))
        vz .+= _vz
    end
    h12 /= sum(pl)
    pl ./= sum(pl)
    vz ./= sum(vz)
    h1 = -sum(filter(isfinite, log2.(vz).*vz))
    h1, -h12
end

"""
Quantify the strength of the interaction between view and place tuning by the percentile of the
real interaction strength with the respect to the interaction when the relationship between view and place was
scrambled.
"""
function estimate_view_by_place_tuning_interaction(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    nn = size(h,1)
    hh = zeros(T, nn)
    hhs = zeros(T, nn)
    for i in 1:nn
        h12,h12s = estimate_view_by_place_tuning_interaction(trialstruct, h[i,:,:], x, y, idxe)
        hh[i] = h12
        hhs[i] = percentile(h12s, 5)
    end
    hh,hhs
end

function estimate_path_length_tuning_2(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, position::AbstractArray{T,3}) where T <: Real
    nt = size(position,3)
    # compute path length from position
    path_length = cumsum([zeros(T,1,nt);sqrt.(dropdims(sum(abs2, diff(position, dims=2),dims=1),dims=1))],dims=1)

    # alternatively, simply 
    # compute occupancy histogram
    hhb = fit(Histogram, path_length[:])
    # compute visited histogram
    hh = fit(Histogram, path_length[:], weights(h[:]))

    hh.edges[1], hh.weights./hhb.weights
end

function get_path_length(trial::RNNTrialStructures.NavigationTrial{T}, position::AbstractArray{T,3};invalid=zero(T)) where T <: Real
     nt = size(position,3)
    path_length = T[]
    idx = CartesianIndex{2}[]
    for i in 1:nt
        idxe = findlast(dropdims(maximum(position[:,:,i],dims=1),dims=1) .> invalid)
        pl = cumsum(sqrt.(dropdims(sum(abs2, diff(position[:,1:idxe,i], dims=2),dims=1),dims=1)),dims=1)
        append!(path_length, pl)
        for j in 2:idxe
            push!(idx, CartesianIndex((j,i)))
        end
    end
    path_length, idx
end

function estimate_path_length_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, position::AbstractArray{T,3},idxe::AbstractVector{Int64}) where T <: Real
    nt = length(idxe)
    path_length = T[]
    hh = T[]
    for i in 1:nt
        pl = cumsum(sqrt.(dropdims(sum(abs2, diff(position[:,1:idxe[i],i], dims=2),dims=1),dims=1)),dims=1)
        append!(path_length, pl)
        append!(hh, h[2:idxe[i],i])
    end
    estimate_path_length_tuning(hh, path_length)
end

function estimate_path_length_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, hh::Vector{T}, path_length::Vector{T}) where T <: Real
    # fit sigmoid
    func(abc) = sum(abs2, scaled_sigmoid.(path_length, abc...) - hh)

    q =optimize(func, ones(T,3))
    yq = scaled_sigmoid.(path_length, q.minimizer...)
    rss = sum(abs2, yq .- hh)
    rss_shuffled = [sum(abs2, yq - shuffle(hh)) for i in 1:1000]
    path_length, hh, q, rss, percentile(rss_shuffled, 1)
end

function estimate_path_length_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Array{T,3}, position::AbstractArray{T,3}) where T <: Real
    path_length, idxf = get_path_length(trialstruct, position)
    tuning_strength = zeros(T, size(h,1))
    params = zeros(T, 3, size(h,1))
    for i in 1:length(tuning_strength)
        pl, hh, q, rss, rss_shuffled = estimate_path_length_tuning(trialstruct, h[i,idxf],path_length)
        params[:,i] = q.minimizer
        tuning_strength[i] = one(T) - rss/rss_shuffled
    end
    tuning_strength, params
end

"""
    get_step_index(y::AbstractArray{T,3}) where T <: Real

Get a flat index over the last 2 dimensions of y with only valid steps
"""
function get_step_index(y::AbstractArray{T,3};invalid=zero(T)) where T <: Real
    nt = size(y,3)
    idx = CartesianIndex{2}[]
    for i in 1:nt
        idxe = findlast(dropdims(maximum(y[:,:,i],dims=1),dims=1) .> invalid)
        for ii in 1:idxe
            push!(idx, CartesianIndex((ii,i)))
        end
    end
    idx
end

function predict_path_length(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, position::AbstractArray{T,3}) where T <: Real
    path_length,idxf = CognitiveSimulations.get_path_length(trialstruct, position)
    X = h[:,idxf]
    train_idx = shuffle(1:length(idxf))[1:round(Int64, 0.8*length(idxf))]
    sort!(train_idx)
    test_idx = setdiff(1:length(idxf),train_idx)
    pca = fit(PCA, X[:,train_idx])
    Xp = predict(pca, X)
    lq = LinearRegressionUtils.llsq_stats(permutedims(Xp[:,train_idx]), repeat(path_length[train_idx],1,1))
    yq = lq.β[1:end-1,:]'*Xp[:,test_idx] .+ lq.β[1,1]
    path_length[test_idx], yq
end