require 'nn'
require 'optim'
require 'modules.rmsprop'
require 'modules.DropoutConstant'

local classic = require 'classic'

local Agent = classic.class('Agent')

function Agent:_init(opt, env_spec, optim_config)
    self.state_space = env_spec.state_space
    self.state_dims = self.state_space[1] * self.state_space[2]
    self.num_actions = env_spec.num_actions

    self.model_spec = opt.model_spec
    self.dropout_p = opt.dropout_p

    -- exploration opts
    self.strategy = {[opt.strategy] = true}
    self.thompson_sampling = {[opt.thompson_sampling] = true}
    -- all
    self.burn_in = opt.burn_in
    -- ε-greedy
    self.eps_start = opt.eps_start
    self.eps_finish = opt.eps_finish
    self.eps_anneal_steps = opt.eps_anneal_steps
    -- dropout
    self.dropout_p_explore = opt.dropout_p_explore
    self.dropout_evaluation_samples = opt.dropout_evaluation_samples
    -- bootstrap dqn
    self.dqns = (self.strategy.thompson_bootstrap or self.strategy.ucb_bootstrap) and opt.dqns or 1
    -- bayesian ucb
    self.ucb_dropout_samples = opt.ucb_dropout_samples
    self.ucb_bootstrap_samples = opt.ucb_bootstrap_samples
    self.ucb_quantile_rate = opt.ucb_quantile_rate
    self.ucb_quantile_sample = opt.ucb_quantile_sample
     
    self.batch_size = opt.batch_size
    self.optim_func = optim.rmspropm
    self.optim_config = {
        learningRate = opt.learning_rate,
        epsilon = opt.rms_eps
    }
    self.gamma = opt.gamma
    self.tderr_clamp = opt.tderr_clamp
    self.gradient_clamp = opt.gradient_clamp

    self.models = {}
    self:create_models()
    self.model = self.models[1].model
end

function Agent:new_episode()
    if self.strategy.thompson_dropout then
        -- sample from posterior
        self:update_dropout_explore_noise(self.model)
    elseif self.strategy.thompson_bootstrap then
        -- sample from posterior
        local k = torch.random(self.dqns)
        self.model = self.models[k].model
    end
end

function Agent:act_train(s_t, step)
    local a_t
    
    -- deactivate dropout layers
    self.model:evaluate()
    
    if step <= self.burn_in then
        -- take random action
        a_t = torch.random(self.num_actions)
    elseif self.strategy.epsilon_greedy then
        -- ε-greedy annealing
        local eps = (self.eps_finish +
                        math.max(0, (self.eps_start - self.eps_finish) * (self.eps_anneal_steps -
                        math.max(0, step - self.burn_in)) / self.eps_anneal_steps))

        if torch.uniform() < eps then
            -- take random action
            a_t = torch.random(self.num_actions)
        else
            -- get greedy action
            a_t = self:get_greedy_action(self.model, s_t)
        end
    elseif self.strategy.thompson_dropout then
        if self.thompson_sampling.step then
            -- sample from posterior each step
            self:update_dropout_explore_noise(self.model)
        end
        -- activate dropout explore layers
        self:toggle_dropout_explore(self.model, true)
        -- get greedy action
        a_t = self:get_greedy_action(self.model, s_t)
    elseif self.strategy.thompson_bootstrap then
        if self.thompson_sampling.step then
            -- sample from posterior
            local k = torch.random(self.dqns)
            self.model = self.models[k].model
            -- deactivate dropout layers
            self.model:evaluate()
        end
        a_t = self:get_greedy_action(self.model, s_t)
    elseif self.strategy.ucb_dropout then
        -- activate dropout explore layers
        self:toggle_dropout_explore(self.model, true)

        -- store q values for each forward pass
        local q = torch.Tensor(self.ucb_dropout_samples, self.num_actions):zero()
        
        -- do T forward passes
        for i = 1, self.ucb_dropout_samples do
            -- sample from posterior
            self:update_dropout_explore_noise(self.model)
            q[i] = self.model:forward(s_t)
        end

        -- get the index of the i'th smallest q value to choose per action
        -- local quantile_index = math.ceil((1 - 1/(self.ucb_quantile_rate*step + 1))*self.ucb_dropout_samples)

        -- sort the q-values in ascending order for each action
        local qsort, _ = torch.sort(q, 1)

        -- act greedily wrt to the quantile
        -- local max_q, max_a = torch.max(qsort[quantile_index],1)
        local max_q, max_a = torch.max(qsort[self.ucb_dropout_samples + 1 - self.ucb_quantile_sample],1)
        a_t = max_a:squeeze()
    elseif self.strategy.ucb_bootstrap then
        -- store q values for each forward pass
        local q = torch.Tensor(self.ucb_bootstrap_samples, self.num_actions):zero()

        -- sample dqns
        local dqn_idxs = torch.randperm(self.dqns):narrow(1,1,self.ucb_bootstrap_samples)

        -- do forward pass for each sample
        for i = 1, dqn_idxs:size(1) do
            -- deactivate dropout layers
            self.models[dqn_idxs[i]].model:evaluate()
            -- get q-values
            q[i] = self.models[dqn_idxs[i]].model:forward(s_t)
        end

        -- get the index of the i'th smallest q value to choose per action
        -- local quantile_index = math.ceil((1 - 1/(self.ucb_quantile_rate*step + 1))*self.ucb_bootstrap_samples)

        -- sort the q-values in ascending order for each action
        local qsort, _ = torch.sort(q, 1)

        -- act greedily wrt to the quantile
        -- local max_q, max_a = torch.max(qsort[quantile_index],1)
        local max_q, max_a = torch.max(qsort[self.ucb_bootstrap_samples + 1 - self.ucb_quantile_sample],1)
        a_t = max_a:squeeze()
    else
        error('no exploration strategy defined')
    end
    return a_t
end

function Agent:act_evaluate(s_t)
    local a_t

    -- deactivate dropout layers
    self.model:evaluate()

    if self.strategy.epsilon_greedy then
        a_t = self:get_greedy_action(self.model, s_t)
    elseif self.strategy.thompson_dropout or self.strategy.ucb_dropout then
        if self.dropout_evaluation_samples > 1 then
            -- average over several posterior samples
            
            -- activate dropout explore layers
            self:toggle_dropout_explore(self.model, true)

            -- store q values for each forward pass
            local q = torch.Tensor(self.dropout_evaluation_samples, self.num_actions):zero()
            
            -- do T forward passes
            for i = 1, self.dropout_evaluation_samples do
                -- sample from posterior
                self:update_dropout_explore_noise(self.model)
                q[i] = self.model:forward(s_t)
            end
            -- get sample mean
            local q_avg = q:mean(1)
            -- act greedily
            local max_q, max_a = torch.max(q_avg, 2)
            a_t = max_a:squeeze()
        else
            -- deactivate dropout explore layers
            self:toggle_dropout_explore(self.model, false)
            a_t = self:get_greedy_action(self.model, s_t)
        end
    elseif self.strategy.thompson_bootstrap or self.strategy.ucb_bootstrap then
        -- frequency count for each action
        local action_count = torch.Tensor(self.num_actions):zero()
        
        for i, model in pairs(self.models) do
            -- deactivate dropout layers
            model.model:evaluate()
            
            -- get argmax_a Q from DQN
            local q = model.model:forward(s_t)
            
            -- Pick an action
            local max_q, max_a = torch.max(q, 2)
            local a = max_a:squeeze()
            
            action_count[a] = action_count[a] + 1
        end
        -- pick action using ensemble method
        local max_count = action_count:max()
        local max_actions = action_count:eq(max_count):nonzero()
        a_t = max_actions[torch.random(max_actions:size(1))]:squeeze()
    else
        error('no exploration strategy defined')
    end
    return a_t
end

function Agent:get_greedy_action(model, s_t)
    local q = model:forward(s_t)
    local max_q, max_a = torch.max(q, 2)
    a_t = max_a:squeeze()
    return a_t
end

function Agent:create_models()
    for i=1, self.dqns do
        local model = self:create_network()
        local model_target = model:clone()
        
        -- remove dropout layers for target network
        model_target:replace(function(module)
           if torch.typename(module) == 'nn.Dropout' then
              return nn.Identity()
           else
              return module
           end
        end)

        local params, grad_params = model:getParameters()
        local params_target, _ = model_target:getParameters()

        self.models[i] = {
            model = model,
            model_target = model_target,
            params = params,
            grad_params = grad_params,
            params_target = params_target,
            optim_state = {}
        }
    end
    log.info('model architecture: ' .. tostring(self.models[1].model))
    log.info('model_target architecture: ' .. tostring(self.models[1].model_target))
end

function Agent:create_network()
    local model = nn.Sequential()
    model:add(nn.View(-1, self.state_dims))
    local last_layer_size = self.state_dims
    for i, n_units in ipairs(self.model_spec) do
        -- add Linear layer and ReLU
        model:add(nn.Linear(last_layer_size, n_units))
        model:add(nn.ReLU(true))

        -- add dropout layer
        if self.dropout_p > 0 then
            model:add(nn.Dropout(self.dropout_p))
        end
        -- add dropoutConstant layer for exploration
        if self.strategy.thompson_dropout or self.strategy.ucb_dropout then
            model:add(nn.DropoutConstant(self.dropout_p_explore))
        end
        
        last_layer_size = n_units
    end
    model:add(nn.Linear(last_layer_size, self.num_actions))

    return model
end

function Agent:update_dropout_explore_noise(model)
    local dropout_modules = model:findModules('nn.DropoutConstant')
    for _, v in pairs(dropout_modules) do
        v:updateNoise()
    end
end

function Agent:toggle_dropout_explore(model, active)
    local dropout_modules = model:findModules('nn.DropoutConstant')
    for _, v in pairs(dropout_modules) do
        v.active = active
    end
end

function Agent:learn_minibatch(train_batch)
    for _, model in pairs(self.models) do
        self:learn_minibatch_model(train_batch, model)
    end
end

function Agent:learn_minibatch_model(train_batch, model)
    local td_err = torch.Tensor(self.batch_size, self.num_actions)

    -- activate dropout layers
    model.model:training()
    
    -- deactivate dropout explore layers
    if self.strategy.thompson_dropout or self.strategy.ucb_dropout then
        self:toggle_dropout_explore(model.model, false)
    end

    -- compute Q values using policy network
    local q = model.model:forward(train_batch.s_t)

    -- use target network to predict q_max
    local q_next = model.model_target:forward(train_batch.s_t1)
    local q_next_max = q_next:max(2):squeeze(2)

    -- check if terminal state
    for b = 1, self.batch_size do
        if train_batch.terminal[b] == 1 then
            q_next[b] = 0
            q_next_max[b] = 0
        end
    end

    -- calculate TD error
    -- zero out the error for actions that we didn't take
    td_err:zero()
    for b = 1, self.batch_size do
        td_err[b][train_batch.a_t[b]] = train_batch.r_t[b] + self.gamma * q_next_max[b] - q[b][train_batch.a_t[b]]
    end
    if self.tderr_clamp > 0 then
        td_err:clamp(-self.tderr_clamp, self.tderr_clamp)
    end

    -- backward pass
    local feval = function(theta)

        -- Reset parameters
        model.grad_params:zero()

        -- Backprop
        model.model:backward(train_batch.s_t, td_err:mul(-2))

        -- Clip Gradients
        if self.gradient_clamp > 0 then
            model.grad_params:clamp(-self.gradient_clamp, self.gradient_clamp)
        end

        return 0, model.grad_params -- don't need to return f(theta)
    end

    -- one rmsprop interation
    self.optim_func(feval, model.params, self.optim_config, model.optim_state)
end

function Agent:update_target_network()
    for _, model in pairs(self.models) do
        model.params_target:copy(model.params)
    end
end

return Agent