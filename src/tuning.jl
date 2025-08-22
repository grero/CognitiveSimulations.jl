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

function estimate_view_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
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

function estimate_view_by_place_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
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

function estimate_view_by_place_tuning_interaction(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    # get view tuning for each place
    vq = estimate_view_by_place_tuning(trialstruct, h, x, y, idxe)

    # estimate overall view tuning
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

function estimate_view_by_place_tuning_interaction(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    nn = size(h,1)
    hh = zeros(T, nn)
    for i in 1:nn
        h1,h12 = estimate_view_by_place_tuning_interaction(trialstruct, h[i,:,:], x, y, idxe)
        hh[i] = h1-h12
    end
    hh
end