using LinearAlgebra
using Makie
using Makie: Colors

plot_theme = theme_minimal()

function plot_network_output(y::AbstractArray{T,3}, ŷ::AbstractArray{T,3},i=1) where T <: Real
    fig = Figure()
    ax1 = Axis(fig[1,1])
    h1 = heatmap!(ax1, permutedims(y[:,:,i]))
    Colorbar(fig[1,2], h1, label="True activity")
    ax2 = Axis(fig[2,1])
    h2 = heatmap!(ax2, permutedims(ŷ[:,:,i]))
    Colorbar(fig[2,2], h2, label="Model activity")
    fig
end

function plot_grid(nrows::Int64, ncols::Int64;kvs...)
    fig = Figure()
    ax = Axis(fig[1,1])
    plot_grid!(ax, nrows, ncols;kvs...)
    fig
end

function plot_grid!(ax, nrows::Int64, ncols::Int64;rowsize=1, colsize=1, origin=(0.0, 0.0))
    points = NTuple{2,Point2f}[]
    for r in 1:nrows+1
        push!(points, (Point2f(0.0, (r-1)*rowsize),Point2f(ncols*colsize, (r-1)*rowsize)))
        #linesegments!(ax, [Point2f(0.0, (r-1)*rowsize)=>Point2f(ncols*colsize, (r-1)*rowsize)])
    end
    for c in 1:ncols+1
        push!(points, (Point2f((c-1)*colsize,0.0),Point2f((c-1)*colsize, nrows*rowsize)))
    end
    porigin = Point2f(origin)
    points = [(p1+porigin, p2+porigin) for (p1,p2) in points] 
    linesegments!(ax, points)
end

"""
Return possible steps
"""
function check_step(i::Int64,j::Int64, ncols::Int64, nrows::Int64) 
    possible_steps = Tuple{Int64, Int64}[]
    if i > 0 
        push!(possible_steps, (-1, 0))
    end
    if i < ncols
        push!(possible_steps, (1, 0))
    end
    if j > 0
        push!(possible_steps, (0,-1))
    end
    if j < nrows
        push!(possible_steps, (0,1))
    end
    possible_steps
end

function do_step!(i::Int64, j::Int64,ncols::Int64, nrows::Int64;Δθ=π/4)
    possible_steps = check_step(i,j,ncols,nrows)
    rand(possible_steps), rand([-Δθ, 0, Δθ])
end

function run_task(::Type{RNNTrialStructures.NavigationTrial}, ncols::Int64, nrows::Int64;colsize=1.0, rowsize=1.0)
    # initial position
    (i,j) = (div(ncols,2), div(nrows,2))
end

function get_view_bins(i::Int64,j::Int64,θ::Real;rowsize=1.0, colsize=1.0)
    θr = [θ-π/4, θ+π/4]
    vp = [cos(θ),sin(θ)]
    wr = rowsize/2
    wc = colsize/2
    pos = ((i-1)*colsize+wc, (j-1)*rowsize+wr)
    x,y = pos

    view_bins = zeros(Bool, 4,10) 
    dl = 0.0001
    v = [cos.(θr) sin.(θr)]
    xq = [x y;x y]
    for i in 1:2
        while all(0.0 .<= xq[i,:] + dl.*v[i,:] .< 5.0)
            xq[i,:] .+= dl.*v[i,:]
        end
    end
    Δ = 1000*eps(Float32)
    horizontal_bins = [extrema(xq[:,1])...,]
    vertical_bins = [extrema(xq[:,2])...,]
    if vp[1] < -Δ && horizontal_bins[1] > Δ
        horizontal_bins[1] = 0.0
    elseif vp[1] > Δ && horizontal_bins[end] < 5.0
        horizontal_bins[end] = 4.9 
    end
    if vp[2] > Δ && vertical_bins[end] < 5.0
        vertical_bins[end] = 4.9 
    elseif vp[2] < -Δ && vertical_bins[1] > Δ
        vertical_bins[1] = 0.0
    end
    
    vb1 = round(Int64, vertical_bins[1]/0.5)+1
    vb2 = round(Int64, vertical_bins[2]/0.5)
    vertical_bin_idx = range(vb1, vb2)
    hb1 = round(Int64,horizontal_bins[1]/0.5)+1
    hb2 = round(Int64, horizontal_bins[2]/0.5)
    horizontal_bin_idx = range(hb1, hb2)
    #vertical_bins[1] = min(y, vertical_bins[1])
    #vertical_bins[2] = max(vertical_bins[2], 5.0)

    # TODO: We do not necessarily always need to touch the sides
    for i in 1:2
        #if (abs(xq[i,1]) <= Δ)# && (vp[1] < -Δ)
        @show xq[i,:]
        if (v[i,1] < -Δ) && Δ .< xq[i,2] < 5.0-Δ
            # left edge 
            @show "left edge"
            view_bins[1,vertical_bin_idx] .= true
        elseif (v[i,1] >= Δ) && ( Δ .<  xq[i,2] .< 5.0-Δ)
        #elseif (abs(xq[i,1]-5.0) <= Δ)# && (vp[1] >= Δ)
            # right edge
            @show "right edge"
            view_bins[3,vertical_bin_idx].= true
        end
        #if (abs(xq[i,2]) <= Δ)# && (vp[2] < -Δ)
        if (v[i,2] < -Δ) && (Δ .< xq[i,1] .< 5.0-Δ)
            # bottom edge
            @show "bottom edge"
            view_bins[2,horizontal_bin_idx].= true

        #elseif (abs(xq[i,2]-5.0) <= Δ)
        elseif (v[i,2] > Δ) && (Δ .< xq[i,1] .< 5.0-Δ)
            # upper edge
            @show "upper edge"
            view_bins[4,horizontal_bin_idx].= true
        end
    end
    view_bins,xq
end

function animate_task(::Type{RNNTrialStructures.NavigationTrial},ncols::Int64, nrows::Int64;rowsize=1.0, colsize=1.0)
    wr = rowsize/2
    wc = colsize/2
    tt = Observable(1)
    # inital position and head direction
    i = 3
    j = 3
    θ = π/2
    ipos = Observable(((i,j), θ))
    on(tt) do _tt
        _ipos = ipos[]
        ((Δi, Δj),Δθ) = do_step!(_ipos[1][1], _ipos[1][2],ncols,nrows)
        ipos[] = ((_ipos[1][1]+Δi, _ipos[1][2]+Δj),_ipos[2]+Δθ)
    end

    pos = lift(ipos) do _ipos
        ((_ipos[1][1]-1)*colsize+wc, (_ipos[1][2]-1)*rowsize+wr)
    end
    θ = lift(ipos) do _pos
        _pos[2]
    end

    view_direction = lift(ipos) do _ipos
        pos = ((_ipos[1][1]-1)*colsize+wc, (_ipos[1][2]-1)*rowsize+wr)
        _θ = _ipos[2]
        xp = cos(_θ)
        yp = sin(_θ)
        
        [(Point2f(pos), Point2f(pos)+Point2f(xp,yp))]
    end

    view_bins_xq = lift(ipos) do ((i,j),θ)
        get_view_bins(i,j,θ;rowsize=rowsize, colsize=colsize)
    end
    view_bins = lift(view_bins_xq) do _vbq
        _vbq[1]
    end

    fov = lift(view_bins_xq,pos) do (_vb, xq), _pos
        [(Point2f(pos), Point2f(xq[1,:])),(Point2f(pos), Point2f(xq[2,:]))]
    end

    fig = Figure()
    ax = Axis(fig[1,1], aspect=1.0)
    plot_grid!(ax, ncols, nrows;rowsize=rowsize, colsize=colsize)
    scatter!(ax, pos, color=:black)
    linesegments!(ax, view_direction, color=:blue)

    #linesegments!(ax, [Point2f(pos)=>Point2f(pos)+Point2f(x1, y1), Point2f(pos)=>Point2f(pos)+Point2f(x2, y2)], color=:green)
    linesegments!(ax, fov, color=:green)

    plot_grid!(ax, 1,10;rowsize=0.5, colsize=0.5, origin=(0.0, -0.75))
    plot_grid!(ax, 1,10;rowsize=0.5, colsize=0.5, origin=(0.0, 5.25))
    plot_grid!(ax, 10,1;rowsize=0.5, colsize=0.5, origin=(-0.75, 0.0))
    plot_grid!(ax, 10,1;rowsize=0.5, colsize=0.5, origin=(5.25, 0.0))

    # indicate filled view bins
    yb = range(0.25, step=0.5, length=10)
    xb = range(0.25, step=0.5, length=10)

    vb_points_left = lift(view_bins) do _vb
        bidx = findall(_vb[1,:])
        if isempty(bidx)
            return [Point2f(NaN)]
        end
        [Point2f(-0.5, yb[ii]) for ii in bidx]
    end

    vb_points_right = lift(view_bins) do _vb
        bidx = findall(_vb[3,:])
        if isempty(bidx)
            return [Point2f(NaN)]
        end
        [Point2f(5.5, yb[ii]) for ii in bidx]
    end

    vb_points_upper = lift(view_bins) do _vb
        bidx = findall(_vb[4,:])
        if isempty(bidx)
            return [Point2f(NaN)]
        end
        [Point2f(xb[ii], 5.5) for ii in bidx]
    end

     vb_points_lower = lift(view_bins) do _vb
        bidx = findall(_vb[2,:])
        if isempty(bidx)
            return [Point2f(NaN)]
        end
        [Point2f(xb[ii], -0.5) for ii in bidx]
    end

    @show vb_points_left
    # left edge
    scatter!(ax, vb_points_left)
    # right edge
    scatter!(ax,vb_points_right) 

    # upper edge
    scatter!(ax, vb_points_upper)
    #bottom edge
    scatter!(ax, vb_points_lower)
    ax2 = Axis(fig[1,2])
    heatmap!(ax2, view_bins)

    @async while true
        tt[] += 1
        sleep(1.0)
        yield()
    end
    display(fig)
    fig
end

function show_task_schematic(::Type{RNNTrialStructures.NavigationTrial};pos=(0.5, 0.5), θ=π/4)
    view_bins = zeros(Bool, 4,10) 
    θr = [θ-π/4, θ+π/4]
    x,y = pos
    d = sqrt(x*x+y*y)
    xp = cos(θ)
    yp = sin(θ)
    vp = [cos(θ),sin(θ)]

    x1 = x*cos(θr[1])/d
    y1 = y*sin(θr[1])/d

    x2 = x*cos(θr[2])/d
    y2 = y*sin(θr[2])/d
    # extend to the edge
    # do the stupid way by just extending along θr until we hit an edge
    dl = 0.0001
    v = [cos.(θr) sin.(θr)]
    xq = [x y;x y]
    horizontal_bins2 = fill(Bool, 2, 10)
    vertical_bins2 = fill(Bool, 2, 10)
    for i in 1:2
        while all(0.0 .<= xq[i,:] + dl.*v[i,:] .< 5.0)
            xq[i,:] .+= dl.*v[i,:]
        end
    end
    Δ = 1000*eps(Float32)
    horizontal_bins = [extrema(xq[:,1])...,]
    vertical_bins = [extrema(xq[:,2])...,]
    if vp[1] < -Δ && horizontal_bins[1] > Δ
        horizontal_bins[1] = 0.0
    elseif vp[1] > Δ && horizontal_bins[end] < 5.0
        horizontal_bins[end] = 4.9 
    end
    if vp[2] > Δ && vertical_bins[end] < 5.0
        vertical_bins[end] = 4.9 
    elseif vp[2] < -Δ && vertical_bins[1] > Δ
        vertical_bins[1] = 0.0
    end
    
    vb1 = round(Int64, vertical_bins[1]/0.5)+1
    vb2 = round(Int64, vertical_bins[2]/0.5)
    vertical_bin_idx = range(vb1, vb2)
    hb1 = round(Int64,horizontal_bins[1]/0.5)+1
    hb2 = round(Int64, horizontal_bins[2]/0.5)
    horizontal_bin_idx = range(hb1, hb2)

    idx0 = round(Int64,max(horizontal_bins[1],  xq[1,1])/0.5)+1
    idx1 = round(Int64, horizontal_bins[end]/0.5)
    @show idx0 idx1

    idx0 = round(Int64,max(horizontal_bins[1],  xq[2,1])/0.5)+1
    idx1 = round(Int64, horizontal_bins[end]/0.5)
    @show idx0 idx1
    #horizontal_bins2[1, idx0:idx1] .= true

    #vertical_bins[1] = min(y, vertical_bins[1])
    #vertical_bins[2] = max(vertical_bins[2], 5.0)

    # TODO: We do not necessarily always need to touch the sides
    for i in 1:2
        @show xq[i,:] v[i,1]
        #if (abs(xq[i,1]) <= Δ)# && (vp[1] < -Δ)
        if (v[i,1] < -Δ) && Δ .< xq[i,2] < 5.0-Δ
            # left edge 
            @show "left edge"
            view_bins[1,vertical_bin_idx] .= true
        elseif (v[i,1] >= Δ) && ( Δ .<  xq[i,2] .< 5.0-Δ)
        #elseif (abs(xq[i,1]-5.0) <= Δ)# && (vp[1] >= Δ)
            # right edge
            @show "right edge"
            view_bins[3,vertical_bin_idx].= true
        end
        #if (abs(xq[i,2]) <= Δ)# && (vp[2] < -Δ)
        if (v[i,2] < -Δ) && (Δ .< xq[i,1] .< 5.0-Δ)
            # bottom edge
            @show "bottom edge"
            view_bins[2,horizontal_bin_idx].= true

        #elseif (abs(xq[i,2]-5.0) <= Δ)
        elseif (v[i,2] > Δ) && (Δ .< xq[i,1] .< 5.0-Δ)
            # upper edge
            @show "upper edge"
            view_bins[4,horizontal_bin_idx].= true
        end
    end
    fig = Figure()
    ax = Axis(fig[1,1], aspect=1.0)
    plot_grid!(ax, 5, 5)
    scatter!(ax, pos, color=:black)
    linesegments!(ax, [Point2f(pos)=>Point2f(pos) + Point2f(xp, yp)], color=:blue)

    #linesegments!(ax, [Point2f(pos)=>Point2f(pos)+Point2f(x1, y1), Point2f(pos)=>Point2f(pos)+Point2f(x2, y2)], color=:green)
    linesegments!(ax, [Point2f(pos)=>Point2f(xq[1,:]), Point2f(pos)=>Point2f(xq[2,:])], color=:green)

    plot_grid!(ax, 1,10;rowsize=0.5, colsize=0.5, origin=(0.0, -0.75))
    plot_grid!(ax, 1,10;rowsize=0.5, colsize=0.5, origin=(0.0, 5.25))
    plot_grid!(ax, 10,1;rowsize=0.5, colsize=0.5, origin=(-0.75, 0.0))
    plot_grid!(ax, 10,1;rowsize=0.5, colsize=0.5, origin=(5.25, 0.0))
    
    # indicate filled view bins
    yb = range(0.25, step=0.5, length=10)
    xb = range(0.25, step=0.5, length=10)

    # left edge
    scatter!(ax, fill(-0.5, sum(view_bins[1,:])), yb[view_bins[1,:]])
    # right edge
    scatter!(ax, fill(5.5, sum(view_bins[3,:])), yb[view_bins[3,:]])

    # upper edge
    scatter!(ax, xb[view_bins[4,:]], fill(5.5, sum(view_bins[4,:])))
    #bottom edge
    scatter!(ax, xb[view_bins[2,:]], fill(-0.5, sum(view_bins[2,:])))
    ax2 = Axis(fig[1,2])
    heatmap!(ax2, view_bins)
    fig
end

function get_view(pos::Vector{T}, θ::T) where T <: Real
    # just project onto a circle which inscribes the arena
    r = sqrt(2*2.5^2)
    pos_center = [2.5, 2.5]
    θr = [θ-π/4, θ+π/4]

    # head direction
    vp = [cos(θ) sin(θ)]

    #fov
    v = [cos.(θr) sin.(θr)]

    # distance to boarder
    dl = 0.01
    xp = fill(0.0, 2, 2)
    for i in 1:2
        xy = [pos...]
        while norm(xy+dl*v[i,:]-pos_center) < r
            xy .+= dl*v[i,:]
        end
        xp[i,:] .= xy
    end

    xq = xp .- pos_center
    θq = atan.(xq[:,2], xq[:,1])
    extrema(θq)
end

function show_task_schematic2(trial::RNNTrialStructures.NavigationTrial;pos=(0.5, 0.5), θ=π/4)
    # just project onto a circle which inscribes the arena
    r = sqrt(2*2.5^2)
    pos_center = [2.5, 2.5]
    θr = [θ-π/4, θ+π/4]

    # head direction
    vp = [cos(θ) sin(θ)]

    #fov
    v = [cos.(θr) sin.(θr)]

    θq1 = RNNTrialStructures.get_view(pos, θ, trial.arena) 
    θq2 = RNNTrialStructures.get_view2(pos, θ, trial.arena) 
    @show θq1 θq2

    fig = Figure()
    ax = Axis(fig[1,1],aspect=1.0)
    plot_grid!(ax, 5, 5)
    lines!(ax, decompose(Point2f, Circle(Point2f(2.5,2.5), r)))
    @show pos
    scatter!(ax, Point2f(pos), color=:black)
    linesegments!(ax, [(Point2f(pos), Point2f(pos)+Point2f(vp[:]))],color=:blue)
    #linesegments!(ax, [(Point2f(pos), Point2f(xp[1,:])),(Point2f(pos), Point2f(xp[2,:]))], color=:green)
    # indicate arc length
    arc!(pos_center, r, extrema(θq1)..., color=:red, linewidth=2.0)
    arc!(pos_center, r, extrema(θq2)..., color=:orange, linewidth=2.0)
    fig
end

label_padding = (-10, 0.0, 0.0, 0.0)
function plot_trial(trial::RNNTrialStructures.NavigationTrial{T},position::Matrix{T}, head_direction::Matrix{T}, viewfield::Matrix{T};tidx=1,Δx=zero(T),sx=one(T)) where T <: Real

    xh = head_direction[:,tidx].*cos.(trial.angular_pref.μ)
    yh = head_direction[:,tidx].*sin.(trial.angular_pref.μ)
    θ = atan(mean(yh),mean(xh))
    v = [cos(θ), sin(θ)]

    fig = Figure()
    ax = Axis(fig[1,1],aspect=1.0,xgridvisible=false, ygridvisible=false)
    Label(fig[1,1,TopLeft()],"A", padding=label_padding)
    plot_grid!(ax,trial.arena.nrows, trial.arena.ncols;rowsize=trial.arena.rowsize, colsize=trial.arena.colsize)
    center_pos = RNNTrialStructures.get_center(trial.arena)
    r = sqrt(sum(abs2, center_pos))
    # also recreate the actual view angles
    pos = position[:,tidx]
    # rescale to the grid
    pos .-= Δx
    pos ./= sx 
    pos .*= RNNTrialStructures.extent(trial.arena)
    θq = RNNTrialStructures.get_view(pos,θ,trial.arena)
    θq = sort([θq...], by=abs)

    r = sqrt(sum(center_pos.^2))
    lines!(ax, decompose(Point2f, Circle(Point2f(center_pos), r)))
    arc!(ax, center_pos, r, θq..., color=:red, linewidth=2.0)

    scatter!(ax, Point2f(pos))
    linesegments!(ax, [(Point2f(pos),Point2f(pos+v))],color=:blue)

    lg = GridLayout(fig[1,2])
    ax2 = Axis(lg[1,1])
    Label(lg[1,1,TopLeft()],"B", padding=label_padding)
    ax2.xticklabelsvisible = false
    h1=heatmap!(ax2, permutedims(viewfield))
    Colorbar(lg[1,2],h1)
    ax3 = Axis(lg[2,1])
    Label(lg[2,1,TopLeft()],"C", padding=label_padding)
    linkxaxes!(ax2, ax3)
    h2 = heatmap!(ax3, permutedims(position))
    Colorbar(lg[2,2],h2)
    ax3.yticks = ([1,2],["x","y"])
    ax3.xlabel = "Time step"
    fig
end

function plot_positions(trial::RNNTrialStructures.NavigationTrial{T}, position::AbstractArray{T,3}, position_true::AbstractArray{T,3},idxe=[size(position,2) for _ in 1:size(position,3)];do_rescale=true) where T <: Real
    eq = RNNTrialStructures.extent(trial.arena)
    if do_rescale
        y = ((position_true .- 0.05)/0.8).*eq
        ŷ = ((position .- 0.05)/0.8).*eq
    else
        y = position_true
        ŷ = position
    end

    fig = Figure()
    ax = Axis(fig[1,1])
    plot_grid!(ax, trial.arena.nrows, trial.arena.ncols;rowsize=trial.arena.rowsize, colsize=trial.arena.rowsize)
    yy = cat([y[:,1:idxe[i],i] for i in 1:size(y,3)]...,dims=2)
    ŷŷ = cat([ŷ[:,1:idxe[i],i] for i in 1:size(y,3)]...,dims=2)
    scatter!(ax, Point2f.(eachcol(yy)))
    scatter!(ax, Point2f.(eachcol(ŷŷ)))
    linesegments!(ax, [(Point2f(_y), Point2f(_ŷ)) for (_y,_ŷ) in zip(eachcol(yy), eachcol(ŷŷ))],color=:gray)
    fig
end

function plot_connectivity_matrix(Wrr::Matrix{T};kwargs...) where T <: Real
    with_theme(plot_theme) do
        fig = Figure()
        lg = GridLayout(fig[1,1])
        plot_connectivity_matrix!(lg, Wrr;kwargs...)
        fig
    end
end

function plot_connectivity_matrix!(lg, Wrr::Matrix{T},x=1:size(Wrr,1), y=1:size(Wrr,2);show_color_label=true, lower_cutoff::Function=x->minimum(x), upper_cutoff::Function=x->maximum(x), do_sort=false, dividers::Union{Vector{<:Real}, Nothing}=nothing, divider_labels::Union{Nothing, Vector{String}}=nothing, show_grouping=true, colorbar_below=true) where T <: Real
    _Wrr = deepcopy(Wrr)
    W_min = lower_cutoff(Wrr)
    W_max = upper_cutoff(Wrr)
    _Wrr[_Wrr .>= W_max] .= W_max
    _Wrr[_Wrr .<= W_min] .= W_min
    ymin,ymax = extrema(filter(isfinite, _Wrr))
    midpoint = abs(ymin)/(ymax-ymin)
    cm = Colors.colormap("RdBu", 100;mid=midpoint)
    if do_sort
        rr = Clustering.affinityprop(Wrr)
        @show sort(unique(rr.assignments))
        sidx = sortperm(rr.assignments)
        _counts = countmap(rr.assignments)
        scounts = cumsum(collect(values(_counts))[sortperm(collect(keys(_counts)))])
    else
        sidx = 1:size(Wrr,1)
        scounts = Int64[]
    end
    with_theme(plot_theme) do
        ax = Axis(lg[1,1])
        hh = heatmap!(ax, x,y,_Wrr[sidx,sidx], colormap=cm)
        if dividers !== nothing
            hlines!(ax, dividers, color=:black, linestyle=:dot)
            vlines!(ax, dividers, color=:black, linestyle=:dot)
            if divider_labels !== nothing
                xt = Float64[]
                for (i,l) in enumerate(divider_labels)
                    if i == 1
                        x0 = dividers[i]/2
                    else
                        x0 = dividers[i-1] + (dividers[i] - dividers[i-1])/2
                    end
                    push!(xt, x0)
                end
                @show xt
                ax.xticks = (xt, divider_labels)
                ax.yticks = (xt, divider_labels)
            end
        end
        if ~isempty(scounts) && show_grouping
            vlines!(ax, scounts, color=:black)
            hlines!(ax, scounts, color=:black)
        end
        if colorbar_below
            _lg = lg[2,1]
            vertical = false
        else
            _lg = lg[1,2]
            vertical = true
        end
        cc = Colorbar(_lg, hh, label="Recurrent weight",vertical=vertical, flipaxis=false, ticks=WilkinsonTicks(3;simplicity_weight=1/6, granularity_weight=2/3, coverage_weight=2/3))
        cc.labelvisible = show_color_label
        ax.xlabel = "Target"
        ax.ylabel = "Source"
        ax
    end
end

function plot_place_field(h::Matrix{T},y::Array{T,3},idxe=[1:size(h,1) for _ in 1:size(h,2)],bins=range(0.0, stop=1.0, step=0.2)) where T <: Real
    nb,nt = size(h)
    hh = fit(Histogram, (y[1,1:idxe[1],1], y[2,1:idxe[1],1]),weights(h[1:idxe[1],1]),(bins,bins))
    for i in 2:nt
        _hh = fit(Histogram, (y[1,1:idxe[i],i], y[2,1:idxe[i],i]),weights(h[1:idxe[i],1]),(bins,bins))
        merge!(hh,_hh)
    end

    fig = Figure()
    ax = Axis(fig[1,1])
    hh = heatmap!(ax, hh.edges[1][1:end-1], hh.edges[2][1:end-1], hh.weights)
    Colorbar(fig[1,2],hh,label="Weight")
    fig
end