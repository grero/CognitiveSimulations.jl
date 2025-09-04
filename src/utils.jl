function StatsBase.mean(X::AbstractArray{T,N}, label::AbstractVector{T2}) where T <: Real where N where T2
    # figure out the axis
    if last(size(X)) !== length(label)
        error("Length of `label` is different from `last(size(X))`")
    end

    ulabel = unique(label)
    newsize = (size(X)[1:end-1]...,length(ulabel))
    nd = ndims(X)
    Y = zeros(T, newsize...)
    cidx = CartesianIndices(tuple([1:size(X,s) for s in 1:nd-1]...))
    for (i,l) in enumerate(ulabel)
        tidx = findall(label.==l)
        
        Y[cidx,i] = dropdims(mean(X[cidx,tidx],dims=nd),dims=nd)
    end
    Y
end