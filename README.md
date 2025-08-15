## Usage

### Random sequence task
Here we train an RNN to solve the random sequence task, roughly following Wang et al (2025). Briefly, the task consist of a random sequence input, where each item in the sequence is drawn randomly from one of 16 fixed angles spanning the circle. During training, the sequence length varies randomly between 2 and 9 from trial to trial. The goal of the task if to report back the sequence as soon as the sequence presentation ends. 

```julia
using CognitiveSimulations
using CognitiveSimulations: train_model
using RecurrentNeuralNetworkModels
using RecurrentNeuralNetworkModels: scaled_tanh
using RNNTrialStructures
using RNNTrialStructures: RandomSequenceTrial, generate_trials, AngularPreference
using StableRNGs
using Makie

# set up input units with tuning curves spanning the circe
apref = AngularPreference(collect(range(0.0f0, stop=2.0f0*π, length=17)), 5.0f0, 0.8f0)

# create a TrialStructure variable describing the task
input_duration = 20.0f0 # the duration of each stimulus presentation
delay_duration = 0.0f0 # no delay between stimulus presentations
go_cue_duration = 20.0f0 # the duration of the go cue
output_duration = 20.0f0 # duration of each step of the output 
min_seq_length = 2
max_seq_length = 9
num_angles # the fixed number of angles to draw each stimulus from
trialstruct = RandomSequenceTrial(input_duration, delay_duration,
                                                     go_cue_duration, outout_duration,
                                                     min_seq_length, max_seq_length,
                                                     num_angles, apref)

# set up an iterator that will deliver a batch of sequences each time it is called
n_seq_per_batch = 10_000
time_step = 20.0f0 # duration of each time step; we want to present each stimulus for just one time step
pre_cue_multiplier = 1.0f0 # the weight to place on the error before go-cue onset
post_cue_multiplier = 1.0f0 # the weight to place on the rror after go-cue onset
rng = StableRNG(1234) # use StableRNG for reproducability
trial_iterator = generate_trials(trialstruct, 10_000, 20.0f0; rseed=UInt32(3), pre_cue_multiplier=1.0f0, post_cue_multiplier=1.0f0, σ=0.0f0, rng=StableRNG(1234))

# now train the model
# we use an RNN with 100 hidden units, with a time constant of 1.0 (i.e. no leak current) and a recurrent noise with ampliude 0.1
n_hidden_units = 100
n_epochs = 20_000
τ=1.0f0 # time constant
η=0.1f0 # recurrent noise amplitude
(ps,st),model = train_model(trial_iterator, 100;nepochs=2_000, performance_aggregator=mean, accuracy_threshold=0.99f0, output_nonlinearity=scaled_tanh, τ=τ,η=η, load_only=false)

```