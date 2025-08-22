module CogSimPlots

using CognitiveSimulations, Makie
using CognitiveSimulations: RNNTrialStructures
using LinearAlgebra
using Makie: Colors
using StatsBase

plot_theme = theme_minimal()

function CognitiveSimulations.plot_network_output(y::AbstractArray{T,3}, ŷ::AbstractArray{T,3},i=1) where T <: Real
    fig = Figure()
    ax1 = Axis(fig[1,1])
    h1 = heatmap!(ax1, permutedims(y[:,:,i]))
    Colorbar(fig[1,2], h1, label="True activity")
    ax2 = Axis(fig[2,1])
    h2 = heatmap!(ax2, permutedims(ŷ[:,:,i]))
    Colorbar(fig[2,2], h2, label="Model activity")
    fig
end

function CognitiveSimulations.plot_grid(nrows::Int64, ncols::Int64;kvs...)
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

function CognitiveSimulations.animate_task(arena::RNNTrialStructures.Arena{T};p_stay=T(0.5),p_hd=T(0.5), fov::T=T(π/3), ntrials=1000) where T <: Real
    tt = Observable(1)
    w,h = RNNTrialStructures.extent(arena)
    x0,y0 = (w/2, h/2)
    r = sqrt((w-x0)^2 + (h-y0)^2)
    # inital position and head direction
    i,j = RNNTrialStructures.get_coordinate(arena)
    θ = Float32(π/2)
    ipos = Observable(((i,j), θ))
    do_pause = Observable(false)
    on(tt) do _tt
        _ipos = ipos[]
        _θ = _ipos[2]
        i,j = _ipos[1]
        Δθ = RNNTrialStructures.get_head_direction(Float32(fov/2), _θ;p_stay=p_stay)
        _θ += Δθ
        (i,j) = RNNTrialStructures.get_coordinate(i,j,arena, _θ;p_hd=p_hd)

        ipos[] = ((i,j),_θ)
    end

    pos = lift(ipos) do _ipos
        i,j = _ipos[1]
        _pos = RNNTrialStructures.get_position(i,j,arena)
        [Point2f(_pos)]
    end

    θ = lift(ipos) do _pos
        _pos[2]
    end

    view_direction = lift(ipos) do _ipos
        i,j = _ipos[1]
        _pos = RNNTrialStructures.get_position(i,j,arena)
        _θ = _ipos[2]
        xp = cos(_θ)
        yp = sin(_θ)
        
        [(Point2f(_pos), Point2f(_pos)+Point2f(xp,yp))]
    end

    #fov = lift(view_bins_xq,pos) do (_vb, xq), _pos
    #    [(Point2f(pos), Point2f(xq[1,:])),(Point2f(pos), Point2f(xq[2,:]))]
    #end

    fov_angles = lift(ipos) do _ipos
        i,j = _ipos[1]
        _θ = _ipos[2]
        _pos = RNNTrialStructures.get_position(i,j, arena)
       _Δθ  = RNNTrialStructures.get_view(_pos, _θ, arena;fov=fov)
       _Δθ
    end

    fov_1 = lift(fov_angles) do _fov
        _fov[1]
    end

    fov_2 = lift(fov_angles) do _fov
        _fov[2]
    end

    fov_lines = lift(ipos) do _ipos
        i,j = _ipos[1]
        _pos = RNNTrialStructures.get_position(i,j, arena)
        _θ = _ipos[2]
        xx1 = RNNTrialStructures.get_circle_intersection([x0,y0], r, _pos, _θ-fov/2)
        xx2 = RNNTrialStructures.get_circle_intersection([x0,y0], r, _pos, _θ+fov/2)
        [(Point2f(_pos), Point2f(xx1)), (Point2f(_pos), Point2f(xx2))]
    end

    fig = Figure()
    ax = Axis(fig[1,1], aspect=1.0)
    circle_points = decompose(Point2f, Circle(Point2f(x0,y0), r))
    lines!(ax, circle_points, color=:black)
    plot_grid!(ax, arena.ncols, arena.nrows;rowsize=arena.rowsize, colsize=arena.colsize)

    scatter!(ax, pos, color=:black)
    linesegments!(ax, view_direction, color=:blue)
    arc!(ax, Point2f(x0,y0), r, fov_1, fov_2, linewidth=2.0, color=:red)

    linesegments!(ax, fov_lines, color=:green)
  
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            # grab the possible steps at the current position
            _ipos = ipos[]
            i,j = _ipos[1]
            possible_steps = RNNTrialStructures.check_step(i,j,arena.ncols,arena.nrows)
            if event.key == Keyboard.p
                do_pause[] = !do_pause[]
            elseif event.key == Keyboard.up
                if (0,1) in possible_steps
                    j += 1
                end
            elseif event.key == Keyboard.down
                if (0,-1) in possible_steps
                    j -= 1
                end
            elseif event.key == Keyboard.left
                if (-1,0) in possible_steps
                    i -= 1
                end
            elseif event.key == Keyboard.right
                if (1,0) in possible_steps
                    i += 1
                end
            end
            ipos[] = ((i,j),_ipos[2])
        end
    end
    @async while tt[] < ntrials 
        if !do_pause[]
            tt[] += 1
            sleep(1.0)
        end
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

label_padding = (-10, 0.0, 0.0, 0.0)
function CognitiveSimulations.plot_trial(trial::RNNTrialStructures.NavigationTrial{T},position::Matrix{T}, head_direction::Matrix{T}, viewfield::Matrix{T};tidx=1,Δx=zero(T),sx=one(T)) where T <: Real

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

function CognitiveSimulations.plot_positions(trial::RNNTrialStructures.NavigationTrial{T}, position::AbstractArray{T,3}, position_true::AbstractArray{T,3},idxe=[size(position,2) for _ in 1:size(position,3)];do_rescale=true) where T <: Real
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

function CognitiveSimulations.plot_connectivity_matrix(Wrr::Matrix{T};kwargs...) where T <: Real
    with_theme(plot_theme) do
        fig = Figure()
        lg = GridLayout(fig[1,1])
        CognitiveSimulations.plot_connectivity_matrix!(lg, Wrr;kwargs...)
        fig
    end
end

function CognitiveSimulations.plot_connectivity_matrix!(lg, Wrr::Matrix{T},x=1:size(Wrr,1), y=1:size(Wrr,2);show_color_label=true, lower_cutoff::Function=x->minimum(x), upper_cutoff::Function=x->maximum(x), do_sort=false, dividers::Union{Vector{<:Real}, Nothing}=nothing, divider_labels::Union{Nothing, Vector{String}}=nothing, show_grouping=true, colorbar_below=true) where T <: Real
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

function CognitiveSimulations.plot_place_field2(h::Matrix{T},y::Array{T,3},idxe=[1:size(h,1) for _ in 1:size(h,2)],bins=range(0.0, stop=1.0, step=0.2)) where T <: Real
    nb,nt = size(h)
    hh = fit(Histogram, (y[1,1:idxe[1],1], y[2,1:idxe[1],1]),weights(h[1:idxe[1],1]),(bins,bins))
    for i in 2:nt
        _hh = fit(Histogram, (y[1,1:idxe[i],i], y[2,1:idxe[i],i]),weights(h[1:idxe[i],i]),(bins,bins))
        merge!(hh,_hh)
    end

    fig = Figure()
    ax = Axis(fig[1,1])
    hh = heatmap!(ax, hh.edges[1][1:end-1], hh.edges[2][1:end-1], hh.weights)
    Colorbar(fig[1,2],hh,label="Weight")
    fig
end

function CognitiveSimulations.plot_place_field(trial::RNNTrialStructures.AbstractTrialStruct{T}, h::Matrix{T},y::Array{T,3},idxe=[1:size(h,1) for _ in 1:size(h,2)],bins=range(0.0, stop=1.0, step=0.2)) where T <: Real
    zp = CognitiveSimulations.estimate_place_tuning(trial, h, y,idxe)
    with_theme(plot_theme) do
        fig = Figure()
        ax = Axis(fig[1,1])
        points = [Point2f(k) for k in keys(zp)]
        vv = collect(values(zp))
        hh = scatter!(ax, points, color=vv, colormap=:Purples, marker=:rect, markersize=0.1, markerspace=:data)
        Colorbar(fig[1,2],hh,label="Weight")
        fig
    end
end

function CognitiveSimulations.plot_place_field(ax::Makie.AbstractAxis, ii::Observable{Int64}, trial::RNNTrialStructures.AbstractTrialStruct{T}, h::AbstractArray{T,3},y::Array{T,3},idxe=[1:size(h,1) for _ in 1:size(h,2)],bins=range(0.0, stop=1.0, step=0.2)) where T <: Real
    zp = lift(ii) do _ii
        CognitiveSimulations.estimate_place_tuning(trial, h[_ii,:,:], y,idxe)
    end

    points = lift(zp) do _zp
        [Point2f(k) for k in keys(_zp)]
    end

    colors = lift(zp) do _zp
        collect(values(_zp))
    end
    scatter!(ax, points, color=colors, colormap=:Purples, marker=:rect, markersize=0.1, markerspace=:data)
    size(h,1)
end

function CognitiveSimulations.plot_place_field(fig::Union{Figure, GridLayout})
    with_theme(plot_theme) do
        ax = Axis(fig[1,1])
        ax
    end
end

function CognitiveSimulations.plot_view_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::Array{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    zp,z, θ = CognitiveSimulations.estimate_view_tuning(trialstruct, h, x, idxe)
    with_theme(plot_theme) do
        fig = Figure()
        ax = PolarAxis(fig[1,1])

        plot_view_tuning!(ax, trialstruct,h,x,idxe)
        fig
    end
end

function CognitiveSimulations.plot_view_tuning(ax::Makie.AbstractAxis, ii::Observable{Int64}, trialstruct, h::AbstractArray{T,3}, x::Array{T,3}, idxe::AbstractVector{Int64};label::Union{Nothing,String}=nothing) where T <: Real
    points = lift(ii) do _ii
        zp,z, θ = CognitiveSimulations.estimate_view_tuning(trialstruct, h[_ii,:,:], x, idxe)
        [Point2f(_θ, _z) for (_θ,_z) in zip(θ,z)]
    end
    on(points) do _points
        reset_limits!(ax)
    end
    lines!(ax, points)
    ax.rticklabelsvisible = false
    ax.rgridvisible = true
    ax.thetagridvisible = true
    size(h,1)
end

function CognitiveSimulations.plot_view_and_hd_tuning(ax::Makie.AbstractAxis, ii::Observable{Int64}, trialstruct, h::AbstractArray{T,3}, x::Array{T,3}, idxe::AbstractVector{Int64};label::Union{Nothing,String}=nothing) where T <: Real
    n = CognitiveSimulations.plot_view_tuning(ax, ii, trialstruct,h,x[1:16,:,:], idxe)
    _ = CognitiveSimulations.plot_view_tuning(ax, ii, trialstruct,h,x[17:end,:,:], idxe)
    n
end

function plot_view_tuning!(ax, trialstruct, h::Matrix{T}, x::Array{T,3}, idxe::AbstractVector{Int64};label::Union{Nothing,String}=nothing) where T <: Real
    zp,z, θ = CognitiveSimulations.estimate_view_tuning(trialstruct, h, x, idxe)

    lines!(ax, θ, z,label=label)
    #lines!(ax, θ, zp)
    ax.rticklabelsvisible = false
    ax.rgridvisible = true
    ax.thetagridvisible = true

end

function CognitiveSimulations.plot_view_and_hd_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::Matrix{T}, x::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    hh = reshape(h, 1, size(h)...)
    CognitiveSimulations.plot_view_and_hd_tuning(1, trialstruct, hh,x,idxe)
end

function CognitiveSimulations.plot_view_and_hd_tuning(ii::Union{Int64, Observable{Int64}}, trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, x::AbstractArray{T,3}, idxe::AbstractVector{Int64};num_axes=2) where T <: Real
    fig,axes = CognitiveSimulations.plot_view_and_hd_tuning(num_axes)
    plot_view_and_hd_tuning!(axes, ii, trialstruct, h, x, idxe)
    #hackish
    if !(typeof(axes) <: NTuple{2})
        axislegend(axes)
    end
    fig
end

function CognitiveSimulations.plot_view_and_hd_tuning(trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, x::AbstractArray{T,3}, idxe::AbstractVector{Int64};num_axes=2) where T <: Real
    fig,axes = CognitiveSimulations.plot_view_and_hd_tuning(num_axes)
    plot_view_and_hd_tuning!(axes, ii, trialstruct, h, x, idxe)
    #hackish
    if !(typeof(axes) <: NTuple{2})
        axislegend(axes)
    end
    fig
end

function CognitiveSimulations.plot_view_and_hd_tuning(fig::Union{Figure, GridLayout};num_axes=1)
    with_theme(plot_theme) do
        ax1 = PolarAxis(fig[1,1])
        axes = ax1
        if num_axes == 2
            ax1.title = "View"
            ax2 = PolarAxis(fig[1,2])
            ax2.title = "Head direction"
            axes = (ax1,ax2)
        end
        axes
    end
end

function CognitiveSimulations.plot_view_tuning(fig::Union{Figure,GridLayout})
    with_theme(plot_theme) do
        ax = PolarAxis(fig[1,1])
    end
end

function plot_view_and_hd_tuning!(axes::NTuple{2, TT}, ii::Int64, trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, x::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real where TT <: Makie.AbstractAxis
    plot_view_tuning!(axes[1], trialstruct, h[ii,:,:], x[1:16,:,:],idxe)
    plot_view_tuning!(axes[2], trialstruct, h[ii,:,:], x[17:end,:,:],idxe)
end

function plot_view_and_hd_tuning!(ax::Makie.AbstractAxis, ii::Int64, trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, x::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    plot_view_tuning!(ax, trialstruct, h[ii,:,:], x[1:16,:,:],idxe;label="View")
    plot_view_tuning!(ax, trialstruct, h[ii,:,:], x[17:end,:,:],idxe;label="Head direction")
end

function CognitiveSimulations.plot_view_by_place_tuning(fig::Union{Figure, GridLayout})
    with_theme(plot_theme) do
        ax = Axis(fig[1,1])
        ax.topspinevisible = true
        ax.rightspinevisible = true
        ax.xticklabelsvisible = false
        ax.yticklabelsvisible = false
        ax
    end
end

function CognitiveSimulations.plot_view_by_place_tuning(ii::AbstractMatrix{<:Union{Int64,Nothing}}, trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    fig = Figure()
    CognitiveSimulations.plot_view_by_place_tuning(fig, ii, trialstruct, h, x, y, idxe)
end

function CognitiveSimulations.plot_view_by_place_tuning(fig::Union{Figure, GridLayout}, ii::AbstractMatrix{<:Union{Int64,Nothing}}, trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    nrows, ncols = size(ii)
    with_theme(plot_theme) do 
        for j in 1:ncols
            for i in 1:nrows
                if ii[i,j] !== nothing
                    ax = Axis(fig[i,j])
                    ax.xticklabelsvisible = false
                    ax.yticklabelsvisible = false
                    ax.topspinevisible = true
                    ax.rightspinevisible = true
                    CognitiveSimulations.plot_view_by_place_tuning(ax, Observable(ii[i,j]), trialstruct, h, x, y, idxe)
                end
            end
        end
        fig
    end
end

function CognitiveSimulations.plot_view_by_place_tuning(ax::Makie.AbstractAxis, ii::Observable{Int64}, trialstruct::RNNTrialStructures.NavigationTrial{T}, h::AbstractArray{T,3}, x::AbstractArray{T,3}, y::AbstractArray{T,3}, idxe::AbstractVector{Int64}) where T <: Real
    # Instead of plotting into separate axes, just draw the tuning curve centered on each of the keys of vq
    # set up colors for each point
    ncols = trialstruct.arena.ncols
    colsize = trialstruct.arena.colsize
    rowsize = trialstruct.arena.rowsize
    nrows = trialstruct.arena.nrows
    grid_pos = Tuple{T,T}[]
    for i in 1:ncols
        for j in 1:nrows
            pp = RNNTrialStructures.get_position(i,j,trialstruct.arena)
            pp = (0.8*pp[1]/(ncols*colsize)+0.05, 0.8*pp[2]/(nrows*rowsize)+0.05)
            push!(grid_pos, pp)
        end
    end
    colors = Observable(zeros(T,length(grid_pos)))
    points = lift(ii) do _ii
        vq = CognitiveSimulations.estimate_view_by_place_tuning(trialstruct, h[_ii,:,:], x, y, idxe)
        _points = Point2f[]
        _colors = zeros(T, length(colors[]))
        for (k,v) in vq
            θ,z = v
            # a bit clunky
            jj = argmin(map(kk->norm(k .- kk), grid_pos))
            _colors[jj] = sum(z)
            # normalize
            z ./= sum(z)
            # scale so that each radius is 0.2
            z .= 0.075*z./maximum(z)
            xx = k[1] .+ z.*cos.(θ)
            yy = k[2] .+ z.*sin.(θ)
            append!(_points, Point2f.(zip(xx,yy)))
            push!(_points, Point2f(NaN))
        end
        colors[] = _colors
        _points
    end
    scatter!(ax, Point2f.(grid_pos),color=colors,colormap=:Purples, markersize=0.05, markerspace=:data)
    lines!(ax, points,color=:black)
    size(h,1)
end


function plot_3d_snapshot!(ax, Z::Array{T,3}, θ::Matrix{T};t::Observable{Int64}=Observable(1),show_trajectories::Observable{Bool}=Observable(false),colormap=:phase)  where T <: Real
     d,nbins,ntrials = size(Z)
    ee = dropdims(mean(sum(abs2.(diff(Z,dims=2)), dims=1),dims=3),dims=(1,3))
    μ = mean(Z, dims=3)
    # random projection matrix
    _W = diagm(fill(one(T), d))
    W = Observable(_W[1:3,1:d])
    rt = Observable(1)
    do_pause = Observable(true)
    R = 0.01*randn(d,d)
    R  = R - permutedims(R) + diagm(fill(one(T),d))
    on(rt) do _rt
        W[] = W[]*R
    end
    # manually assign colors so that we can use them for the trajectories as well
    k = 1
    acolors = resample_cmap(colormap, size(θ,k))
    sidx = sortperm(θ[:,k])
    vidx = invperm(sidx)
    pcolors = Observable(acolors[vidx])
    points = lift(t,W) do _t, _W
        Point3f.(eachcol(_W*(Z[:,_t, :] .- μ[:,_t,:])))
    end
    traj = lift(t,W) do _t, _W
        [_t >= i >= 1 ? Point3f(_W*(Z[:, i, j] - μ[:,_t])) : Point3f(NaN) for j in 1:size(Z,3) for i in (_t-5):_t+1]
    end
    traj_color = lift(pcolors) do _pc
         [_pc[j] for j in 1:size(θ,1) for i in 1:7]
    end
    scatter!(ax, points, color=pcolors)
    ll = lines!(ax, traj, color=traj_color)
    ll.visible = show_trajectories[]
    on(show_trajectories) do _st
        ll.visible = _st
    end
end

function CognitiveSimulations.plot_3d_snapshot(Z::Array{T,3},θ::Matrix{T},timepoints::Vector{Int64},angle_idx=fill(1,length(timepoints));show_tickabels=false, show_axes_labels=false, kwargs...) where T <: Real
    fig = Figure()
    #axes = [Axis3(fig[1,i]) for i in 1:length(timepoints)]
    for (i,(tp,ia)) in enumerate(zip(timepoints,angle_idx))
        if length(ia) == 1
            if ia == 1
                cm = :phase
            else
                cm = :seaborn_icefire_gradient
            end
            ax = Axis3(fig[1,i])
            plot_3d_snapshot!(ax, Z, θ[:,ia:ia];t=Observable(tp),colormap=cm, kwargs...)
            if !show_tickabels
                ax.xticklabelsvisible = false
                ax.yticklabelsvisible = false
                ax.zticklabelsvisible = false
            end
            if !show_axes_labels
                ax.xlabelvisible = false
                ax.ylabelvisible = false
                ax.zlabelvisible = false
            end
        else
            lg = GridLayout(fig[1,i])
            axes = [Axis3(lg[j,1]) for j in 1:length(ia)]
            for (j,(ax,_ia)) in enumerate(zip(axes, ia))
                if _ia == 1
                    cm = :phase
                else
                    cm = :seaborn_icefire_gradient
                end
                plot_3d_snapshot!(ax, Z, θ[:,_ia:_ia];t=Observable(tp),colormap=cm, kwargs...)
                if !show_tickabels
                    ax.xticklabelsvisible = false
                    ax.yticklabelsvisible = false
                    ax.zticklabelsvisible = false
                end
                if !show_axes_labels
                    ax.xlabelvisible = false
                    ax.ylabelvisible = false
                    ax.zlabelvisible = false
                end
            end
            if i == length(timepoints) # only show color bar for the last plot
                lgc = GridLayout(fig[1,i+1])
                for (j,_ia) in enumerate(ia)
                    if _ia == 1
                        cm = :phase
                    else
                        cm = :seaborn_icefire_gradient
                    end
                    Colorbar(lgc[j,1],limits=(minimum(θ), maximum(θ)), colormap=cm, label="θ$(_ia)")
                end
            end
            rowgap!(lg, 1, 0.1)
        end
    end
    # add labels
    for (i,tp) in enumerate(timepoints)
        Label(fig[1,i,Top()], "Time point $tp")
    end
    fig
end

function CognitiveSimulations.plot_3d_snapshot(Z::Array{T,3}, θ::Matrix{T};t::Observable{Int64}=Observable(1),show_trajectories::Observable{Bool}=Observable(false), trial_events::Vector{Int64}=Int64[], fname::String="snapshot.png",colormap=:phase) where T <: Real
    is_saving = Observable(false)
    d,nbins,ntrials = size(Z)
    ee = dropdims(mean(sum(abs2.(diff(Z,dims=2)), dims=1),dims=3),dims=(1,3))
    μ = mean(Z, dims=3)
    # random projection matrix
    _W = diagm(fill(one(T), d))
    W = Observable(_W[1:3,1:d])
    rt = Observable(1)
    do_pause = Observable(true)
    R = 0.01*randn(d,d)
    R  = R - permutedims(R) + diagm(fill(one(T),d))
    on(rt) do _rt
        W[] = W[]*R
    end
    # manually assign colors so that we can use them for the trajectories as well
    k = 1
    acolors = resample_cmap(colormap, size(θ,k))
    sidx = sortperm(θ[:,k])
    vidx = invperm(sidx)
    pcolors = Observable(acolors[vidx])
    points = lift(t,W) do _t, _W
        Point3f.(eachcol(_W*(Z[:,_t, :] .- μ[:,_t,:])))
    end
    traj = lift(t,W) do _t, _W
        [_t >= i >= 1 ? Point3f(_W*(Z[:, i, j] - μ[:,_t])) : Point3f(NaN) for j in 1:size(Z,3) for i in (_t-5):_t+1]
    end
    traj_color = lift(pcolors) do _pc
         [_pc[j] for j in 1:size(θ,1) for i in 1:7]
    end

    # if show trajectories, include fading trajectories of the last 5 points
    fig = Figure()
    ax = Axis3(fig[1,1])
    cax = Colorbar(fig[1,2], limits=(minimum(θ), maximum(θ)), colormap=:phase)
    cax.label = "θ1"
    scatter!(ax, points, color=pcolors)
    ll = lines!(ax, traj, color=traj_color)
    ll.visible = show_trajectories[]
    on(show_trajectories) do _st
        ll.visible = _st
    end
    tt = textlabel!(ax, 0.05, 0.05, text="c : rotate color axis\nr : change projection\np : rotate projection\nt : toggle traces", space=:relative,
              background_color=:black, alpha=0.2, text_align=(:left, :bottom))
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            if event.key == Keyboard.left
                t[] = max(0, t[]-1)
            elseif event.key == Keyboard.right
                t[] = min(size(Z,2), t[]+1)
            elseif event.key == Keyboard.r
                q,r = qr(randn(d,d))
                W[] = permutedims(q[:,1:3])
            elseif event.key == Keyboard.p
                do_pause[] = !do_pause[]
            elseif event.key == Keyboard.c
                k = mod(k,size(θ,2))+1
                sidx = sortperm(θ[:,k])
                vidx = invperm(sidx)
                pcolors[] = acolors[vidx]
                cax.label = "θ$k"
            elseif event.key == Keyboard.t
                show_trajectories[] = !show_trajectories[]
            elseif event.key == Keyboard.s
                is_saving[] = true
                bn,ex = splitext(fname)
                _fname = replace(fname, ex => "_$(t[])$(ex)")
                save(_fname, fig;px_per_unit=8)
                is_saving[] = false
            end
        end
        #autolimits!(ax)
    end
    # show the average enery
    axe = Axis(fig[2,1])
    lines!(axe, 2:length(ee)+1, ee, color=:black)
    if !isempty(trial_events)
        vlines!(axe, trial_events, color=Cycled(1))
    end
    vlines!(axe, t, color=:black, linestyle=:dot)

    axe.ylabel = "Avg speed"
    axe.xticklabelsvisible = false
    axe.xgridvisible = false
    axe.ygridvisible = false
    axe.topspinevisible = false
    axe.rightspinevisible = false
    rowsize!(fig.layout, 2, Relative(0.2))
    sl = Slider(fig[3,1], range=range(1, stop=size(Z,2), step=1), startvalue=t[], update_while_dragging=true)

    on(t) do _t
        _min,_max = extrema(Z[:,t[], :] .- μ[:,t[],:])
        _mm = maximum(abs.([_min, _max]))
        Δ = 2*_mm
        _min = -_mm - 0.15*Δ
        _max = _mm + 0.15*Δ
        xlims!(ax, _min, _max)
        ylims!(ax, _min, _max)
        zlims!(ax, _min, _max)
    end

    on(is_saving) do _is_saving
        tt.visible[] = !_is_saving
        sl.blockscene.visible[] = !_is_saving
    end

    on(sl.value) do _v
        if t[] != _v
            t[] = _v
        end
    end

    on(t) do _t
        if sl.value[] != _t
            set_close_to!(sl, _t)
        end
    end
    @async while true
        if !do_pause[]
            rt[] = rt[] + 1
        end
        sleep(0.1)
        yield()
    end
    fig
end

end
