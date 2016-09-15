local utils = {}

function utils.tprint(tbl, output_fn, indent)
    indent = indent or 0
    output_fn = output_fn or print
    for k, v in pairs(tbl) do
        formatting = string.rep("  ", indent) .. k .. ": "
        if type(v) == "table" then
            output_fn(formatting)
            utils.tprint(v, output_fn, indent+1)
        elseif type(v) == 'boolean' then
            output_fn(formatting .. tostring(v))      
        else
            output_fn(formatting .. v)
        end
    end
end

return utils