function plot_network_output(y::AbstractArray{T,3}, ŷ::AbstractArray{T,3},i=1) where T <: Real
    fig = Figure()
    ax1 = Axis(fig[1,1])
    h = heatmap!(ax1, permutedims(y[:,:,i]))
    Colorbar(fig[1,2], h, label="Activity")
    ax1 = Axis(fig[2,1])
    h = heatmap!(ax1, permutedims(ŷ[:,:,i]))
    Colorbar(fig[2,2], h, label="Activity")
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