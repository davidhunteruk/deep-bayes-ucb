local display = require 'display'
local image = require 'image'
local env = (require 'game.GridWorldPixels')()

local keyboard_map = {
    w = 1,
    d = 2,
    s = 3,
    a = 4
}

env:start()
local game_win = display.image(env:getDisplay())

-- 4 actions: 1=up, 2=right, 3=down, 4=left
function act(action)
    -- os.execute('sleep 0.1')
    local r_t, s_t1, terminal = env:step(action)
    display.image(env:getDisplay(), {win=game_win})
    print(action, r_t, terminal)
    return r_t, terminal
end

function manual()
    env:start()
    display.image(env:getDisplay(), {win=game_win})
    local exit = false
    while not exit do
        input = io.read()
        if input == 'q' then
            exit = true
        elseif input == 'e' then
            env:start()
            display.image(env:getDisplay(), {win=game_win})
        else
            act(keyboard_map[input])
        end
    end
end

function random()
    env:start()
    display.image(env:getDisplay(), {win=game_win})
    local terminal = false
    local i = 0
    while i < 1000 do
        _, terminal = act(torch.random(4))
        i = i + 1
    end
    print(i)
end

manual()
-- random()


-- for i = 1,6 do
--     act(3)
-- end

-- for i = 1,6 do
--     act(2)
-- end

-- for i = 1,6 do
--     act(1)
-- end

-- for i = 1,6 do
--     act(4)
-- end


-- function scale(img, factor)
--     local factor = factor or 50
--     return image.scale(img, s_t:size(1)*32, s_t:size(2)*32, 'simple')
-- end

-- env = (require 'mazebase-rlenv.MazeBaseRLEnv')()
-- s_t = env:start()
-- local game_win_simple = display.image(scale(s_t))
-- local game_win = display.image(env:getDisplay())

-- for i = 1,1000 do
--     os.execute('sleep 0.01')
--     local r_t, s_t1, terminal = env:step(torch.random(4))
--     display.image(env:getDisplay(), {win=game_win})
--     display.image(scale(s_t1), {win=game_win_simple})
--     print(i, r_t, terminal)
-- end