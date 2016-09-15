log = require 'include.log'
local display = require 'display'
local utils = require 'include.utils'


-- CONFIGURATION
-- -------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('DQN GridWorld')
cmd:text()
cmd:text('Options')

-- id for logging purposes
cmd:option('-exp_id', 0, 'experiment id')

-- computation
cmd:option('-seed', 5, 'initial random seed')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-cuda', 0, 'cuda')

-- model
cmd:option('-gamma', 0.9, 'discount factor')
cmd:option('-replay_memory', 1e+5, 'experience replay memory')

-- exploration
cmd:option('-strategy', 'ucb_bootstrap', 'exploration strategy: epsilon_greedy|thompson_dropout|thompson_bootstrap|ucb_dropout|ucb_bootstrap')
cmd:option('-burn_in', 500, 'number of random actions to take at start')
cmd:option('-eps_start', 1, 'start ε-greedy policy')
cmd:option('-eps_finish', 0.1, 'final ε-greedy policy')
cmd:option('-eps_anneal_steps', 10000, 'number of steps to anneal ε')
cmd:option('-thompson_sampling', 'episode', 'when to sample from posterior step|episode')
cmd:option('-dropout_p_explore', 0.15, 'dropout probability for exploration')
cmd:option('-dropout_evaluation_samples', 0, 'number of posterior samples to average for evaluation')
cmd:option('-dqns', 5, 'number of bootstrapped DQNs')
cmd:option('-ucb_bootstrap_samples', 2, 'number of posterior samples for ucb_bootstrap')
cmd:option('-ucb_dropout_samples', 10, 'number of posterior samples for ucb_dropout')
cmd:option('-ucb_quantile_rate', 0.001, 'controls rate of increase of quantile wrt t')
cmd:option('-ucb_quantile_sample', 1, 'the highest x sample to take for each action')


-- training
cmd:option('-steps', 200000, 'number of training steps')
cmd:option('-max_episode_steps', 50, 'max number of steps per episode')
cmd:option('-batch_size', 32, 'batch size')
cmd:option('-learning_rate', 1e-3, 'learning rate')
cmd:option('-rms_eps', 1e-6, 'rmsprop epsilon')
cmd:option('-gradient_clamp', 3, 'clamp NN gradients. Set to 0 to deactivate')
cmd:option('-tderr_clamp', 1, 'clamp TD error. Set to 0 to deactivate')
cmd:option('-dropout_p', 0, 'dropout probability for training' )
cmd:option('-target_update_freq', 50, 'target network update frequency (steps)')
cmd:option('-duplicate_actions_halt', 0, 'number of test duplicate action sequences before halting')
cmd:option('-max_reward', 1.950, 'maximum possible reward that can be achieved in an episode')
cmd:option('-max_reward_seq_halt', 5, 'number of consecutive max reward test runs before halting')

-- output
cmd:option('-test_frequency', 100, 'test frequency (steps)')
cmd:option('-display_game', false, 'display game')
cmd:option('-logfile', true, 'log to log file')

cmd:text()

local opt = cmd:parse(arg)

-- default model spec
opt.model_spec = {25,50}

-- experiment overrides
if exp then
    for k,v in pairs(exp) do opt[k] = v end
end


-- INITIALISATION
-- --------------

-- initialise logs
if opt.logfile then
    log.outfile = paths.concat('experiments', 'id' .. opt.exp_id ..'_seed' .. opt.seed .. '.log')
end

-- log options
log.info('opt:')
utils.tprint(opt, log.info, 1)

-- initialise randomisers
math.randomseed(opt.seed)
torch.manualSeed(opt.seed)

-- initialise computation
torch.setnumthreads(opt.threads)

-- Tensor creation function for removing need to cast to CUDA if GPU is enabled
Tensor = torch.Tensor
if opt.cuda > 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.cuda)
    -- Set manual seeds using random numbers to reduce correlations
    cutorch.manualSeed(torch.random())
    print(cutorch.getDeviceProperties(opt.cuda))
    Tensor = torch.CudaTensor
end

makecuda = function(thing)
  if opt.cuda > 0 then
    return thing:cuda()
  end
  return thing
end

-- initialise environment
local env = (require 'game.GridWorldPixels')(opt)
local env_spec = {
    num_actions = env:getActionSpec()[3][2], -- number of actions
    state_space = env:getStateSpec()[2] -- {grid_height, grid_width}
}
log.info('env_spec:')
utils.tprint(env_spec, log.info, 1)

-- initialise agent
local agent = (require 'Agent')(opt, env_spec)

-- minibatch storage
local minibatch = {
    s_t = torch.Tensor(opt.batch_size, env_spec.state_space[1], env_spec.state_space[2]),
    a_t = torch.Tensor(opt.batch_size),
    r_t = torch.Tensor(opt.batch_size),
    s_t1 = torch.Tensor(opt.batch_size, env_spec.state_space[1], env_spec.state_space[2]),
    terminal = torch.Tensor(opt.batch_size)
}

-- initialise display
local game_win = nil
local function display_game()
    if opt.display_game then
        game_win = display.image(env:getDisplay(), {win=game_win})
        os.execute('sleep 0.01')
    end
end

-- initialise experiment output
output = {}
output.opt = opt
output.test = {}


-- MAIN TRAINING LOOP
-- ------------------

local step = 1
local episode = {
    terminal = true
}
local replay_mem = {}
local duplicate_actions_count = 0
local last_test_actions = ''
local max_reward_seq = 0
local halt = false

while step <= opt.steps and not halt do

    if step % opt.test_frequency == 0 then
        episode.display_next = true
    end

    -- new episode: reset environment and get first state
    if episode.terminal or episode.t > opt.max_episode_steps then
        episode.t = 1
        episode.s_t = env:start():clone()
        episode.terminal = false
        episode.r_total = 0

        -- notify agent
        agent:new_episode(e)

        episode.display = false
        if episode.display_next then
            episode.display_next = false
            episode.display = true
            display_game()
        end
    end

    -- get action from agent
    episode.a_t = agent:act_train(episode.s_t, step)

    -- sample transition from environment
    episode.r_t, episode.s_t1, episode.terminal = env:step(episode.a_t)
    episode.s_t1 = episode.s_t1:clone()
    episode.r_total = episode.r_total + episode.r_t

    if episode.display then
        display_game()
    end

    -- add transition to replay memory
    local r_id = ((step-1) % opt.replay_memory) + 1
    replay_mem[r_id] = {
        s_t = episode.s_t,
        a_t = episode.a_t,
        r_t = episode.r_t,
        s_t1 = episode.s_t1,
        terminal = episode.terminal and 1 or 0
    }

    -- update Q network with minibatch
    if #replay_mem >= opt.batch_size then
        
        -- create minibatch
        for b = 1, opt.batch_size do
            local exp_id = torch.random(#replay_mem)
            minibatch.s_t[b] = replay_mem[exp_id].s_t
            minibatch.a_t[b] = replay_mem[exp_id].a_t
            minibatch.r_t[b] = replay_mem[exp_id].r_t
            minibatch.s_t1[b] = replay_mem[exp_id].s_t1
            minibatch.terminal[b] = replay_mem[exp_id].terminal
        end
        
        -- update Q network
        agent:learn_minibatch(minibatch)
        
        -- update target network
        if step % opt.target_update_freq == 0 then
            agent:update_target_network()
            collectgarbage()
        end
    end

    if step % opt.test_frequency == 0 then
        -- Test agent
        local s_t, a_t, r_t, s_t1
        local terminal = false

        -- Initial state
        s_t = env:start():clone()
        -- display_game()

        local t = 1
        local r_total = 0
        local actions = ''
        while t <= opt.max_episode_steps and not terminal do

            a_t = agent:act_evaluate(s_t, false)
            
            actions = actions .. a_t

            --compute reward for current state-action pair
            r_t, s_t1, terminal = env:step(a_t)
            s_t1 = s_t1:clone()
            r_total = r_total + r_t

            -- display_game()

            -- next state
            s_t = s_t1:clone()
            t = t + 1
        end

        table.insert(output.test, {
            steps = step,
            reward = r_total,
            actions = actions
        })

        output.last_model = agent.models
        log.infof('steps=%d,  test_r=%.3f, actions=%s', step, r_total, actions)

        if actions == last_test_actions then
            duplicate_actions_count = duplicate_actions_count + 1
            if opt.duplicate_actions_halt > 0 and duplicate_actions_count >= opt.duplicate_actions_halt then
                halt = true
            end
        else
            duplicate_actions_count = 0
        end
        last_test_actions = actions

        if opt.max_reward_seq_halt >= 0 and r_total >= opt.max_reward then
            max_reward_seq = max_reward_seq + 1
            if max_reward_seq >= opt.max_reward_seq_halt then
                halt = true
            end
        else
            max_reward_seq = 0
        end
    end

    -- next step
    episode.s_t = episode.s_t1:clone()
    episode.t = episode.t + 1
    step = step + 1
end











