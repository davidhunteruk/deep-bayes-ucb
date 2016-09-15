local classic = require 'classic'
local image = require 'image'

local GridWorldPixels = classic.class('GridWorldPixels')

function GridWorldPixels:_init(opt)
    opt = opt or {}
    self.width = 7
    self.height = 7
    -- format {y, x, pixel_intensity}
    self.agent = {1, 1, opt.agent_intensity or 1}
    -- format {y, x, pixel_intensity}
    self.goals = {
        {7, 7, opt.goal1_intensity or 0.25},
        {1, 7, opt.goal2_intensity or 0.5},
        {7, 1, opt.goal3_intensity or 0.75}
    }
    self.rewards = {
        goal = 0.25,
        bonus12 = 0.5,
        bonus123 = 1,
        action = opt.action_reward or -0.01
    }
    self.agent_pos = {}
    self.goals_visited = 0
    self.displayZoom = 30
    self.background_intensity = opt.background_intensity or 0
    self.pixel_representation = torch.Tensor(self.height, self.width)
end

-- pixel-based representation
function GridWorldPixels:getStateSpec()
    return {'real', {self.height, self.width}, {0, 1}}
end

-- 4 actions: 1=up, 2=right, 3=down, 4=left
function GridWorldPixels:getActionSpec()
    return {'int', 1, {1, 4}}
end

-- min and max reward
function GridWorldPixels:getRewardSpec()
    return {-0.1, 5}
end

function GridWorldPixels:start()
    self.agent_pos[1] = self.agent[1]
    self.agent_pos[2] = self.agent[2]
    for _, v in pairs(self.goals) do
        v.order = 0
    end
    self.goals_visited = 0
    self:updatePixelRepresentation()
    return self.pixel_representation
end

function GridWorldPixels:step(action)
    local terminal = false
    local r = self.rewards.action

    -- Move
    if action == 1 then
        -- Move up
        self.agent_pos[1] = math.max(self.agent_pos[1] - 1, 1)
    elseif action == 2 then
        -- Move right
        self.agent_pos[2] = math.min(self.agent_pos[2] + 1, self.width)
    elseif action == 3 then
        -- Move down
        self.agent_pos[1] = math.min(self.agent_pos[1] + 1, self.height)
    else
        -- Move left
        self.agent_pos[2] = math.max(self.agent_pos[2] - 1, 1)
    end

    for k, v in pairs(self.goals) do
        if v.order == 0 and self.agent_pos[1] == v[1] and self.agent_pos[2] == v[2] then
            self.goals_visited = self.goals_visited + 1
            v.order = self.goals_visited
            r = r + self.rewards.goal

            if k == 2 and ((v.order == 2 and self.goals[1].order == 1) or (v.order == 3 and self.goals[1].order == 2)) then
                r = r + self.rewards.bonus12
            elseif k == 3 and v.order == 3 and self.goals[2].order == 2 and self.goals[1].order == 1 then
                r = r + self.rewards.bonus123
            end

            if self.goals_visited == 3 then terminal = true end
        end
    end

    self:updatePixelRepresentation()

    return r, self.pixel_representation, terminal
end

function GridWorldPixels:updatePixelRepresentation()
    self.pixel_representation:fill(self.background_intensity)
    
    for _, v in pairs(self.goals) do
        if v.order == 0 then
            self.pixel_representation[v[1]][v[2]] = v[3]
        end
    end

    self.pixel_representation[self.agent_pos] = self.agent[3]
end

function GridWorldPixels:getDisplaySpec()
    return {'real', {self.height*self.displayZoom}, {self.width*self.displayZoom}, {0,1}}
end

function GridWorldPixels:getDisplay()
    return image.scale(self.pixel_representation, self.height*self.displayZoom, self.width*self.displayZoom, 'simple')
end

return GridWorldPixels