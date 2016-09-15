local classic = require 'classic'
local image = require 'image'

local GridWorldPixelsRandom = classic.class('GridWorldPixelsRandom')

function GridWorldPixelsRandom:_init()
    self.width = 5
    self.height = 5
    -- format {y, x, pixel_intensity}
    self.agent = {3, 3, 1.0}
    -- format {y, x, pixel_intensity}
    self.bomb = {
        intensity = -0.5,
        pos = {3,3}
    }
    self.goal = {
        intensity = -1.0,
        pos = {3,3}
    }
    self.rewards = {
        goal = 1.1,
        bomb = -0.9,
        action = -0.1
    }
    self.agent_pos = {}
    self.displayZoom = 30
    self.pixel_representation = torch.Tensor(self.height, self.width)
end

-- pixel-based representation
function GridWorldPixelsRandom:getStateSpec()
    return {'real', {self.height, self.width}, {0, 1}}
end

-- 4 actions: 1=up, 2=right, 3=down, 4=left
function GridWorldPixelsRandom:getActionSpec()
    return {'int', 1, {1, 4}}
end

-- min and max reward
function GridWorldPixelsRandom:getRewardSpec()
    return {-1, 1}
end

function GridWorldPixelsRandom:start()
    self.agent_pos[1] = self.agent[1]
    self.agent_pos[2] = self.agent[2]
    self.goal.pos = self:getRandomFreePosition()
    self.bomb.pos = self:getRandomFreePosition()
    self:updatePixelRepresentation()
    return self.pixel_representation
end

function GridWorldPixelsRandom:step(action)
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

    if self.agent_pos[1] == self.goal.pos[1] and self.agent_pos[2] == self.goal.pos[2] then
        r = r + self.rewards.goal
        self.goal.pos = self:getRandomFreePosition()
    elseif self.agent_pos[1] == self.bomb.pos[1] and self.agent_pos[2] == self.bomb.pos[2] then
        r = r + self.rewards.bomb
        self.bomb.pos = self:getRandomFreePosition()
    end

    self:updatePixelRepresentation()

    return r, self.pixel_representation, terminal
end

function GridWorldPixelsRandom:getRandomFreePosition()
    local pos = {}
    local conflict = true
    while conflict do
        pos = {
            torch.random(self.width),
            torch.random(self.height)
        }
        if (pos[1] ~= self.agent_pos[1] or pos[2] ~= self.agent_pos[2]) and
           (pos[1] ~= self.goal.pos[1] or pos[2] ~= self.goal.pos[2]) and
           (pos[1] ~= self.bomb.pos[1] or pos[2] ~= self.bomb.pos[2]) then
            conflict = false
        end
    end
    return pos
end

function GridWorldPixelsRandom:updatePixelRepresentation()
    self.pixel_representation:fill(0)
    self.pixel_representation[self.agent_pos] = self.agent[3]
    self.pixel_representation[self.goal.pos] = self.goal.intensity
    self.pixel_representation[self.bomb.pos] = self.bomb.intensity
end

function GridWorldPixelsRandom:getDisplaySpec()
    return {'real', {self.height*self.displayZoom}, {self.width*self.displayZoom}, {0,1}}
end

function GridWorldPixelsRandom:getDisplay()
    return image.scale(self.pixel_representation, self.height*self.displayZoom, self.width*self.displayZoom, 'simple')
end

return GridWorldPixelsRandom