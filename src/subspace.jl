"""
    get_subspace(X::AbstractVector{T,3}, θ::Matrix{T2}, idx0::Int64) where T <: Real where T2

Find an orthongal subspace containing information about θ at time point `idx0`.
"""
function get_subspace(X::AbstractArray{T,3}, θ::Matrix{T2}, idx0::Int64) where T <: Real where T2
    Y = dropdims(mean(X[:,idx0:idx0+1, :],dims=2),dims=2)   
    _,w12 = MultiDimensionalTimeSeriesPlots.rpca(Y, eachcol(MultiDimensionalTimeSeriesPlots.Angle.(θ))...)
    Z12 = mapslices(x->w12*x, X, dims=1)
    Z12, w12
end

function get_subspace(trialstruct, training_args, testing_args=training_args;batch_size=256, align_event=trialstruct.response_onset[1]-2, kwargs...)
    # get a model
    (ps,st), model, _ = train_model(trialstruct, batch_size, training_args...) 
    trial_iterator = RNNTrialStructures.generate_trials(trialstruct, 1024;testing_args...)
    x,y,w = trial_iterator()
    θ = RNNTrialStructures.readout(trialstruct, y) 
    (ŷ,h), ps2 = model(x, ps, RecurrentNetworkModels.Lux.testmode(st))
    perf = RNNTrialStructures.performance(trialstruct, ŷ, y)
    Z2,w2 = CognitiveSimulations.get_subspace(h, θ,align_event) 
    Z2,w2,perf
end