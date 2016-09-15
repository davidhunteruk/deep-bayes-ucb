local DropoutConstant, Parent = torch.class('nn.DropoutConstant', 'nn.Module')

--[[
Target network shouldn't use dropout
Policy network should only use dropout when training, it should not use dropout on batch update or testing
]]--

function DropoutConstant:__init(p)
   Parent.__init(self)
   self.p = p
   self.active = false
   self.noise = torch.Tensor()
end

function DropoutConstant:updateOutput(input)   
   if self.active then
      self.output:resizeAs(input):copy(input)
      if self.noise:nDimension() == 0 then
         self.noise:resizeAs(input)
         self.noise:bernoulli(1-self.p)
         self.noise:div(1-self.p)
      end
      self.output:cmul(self.noise)
   else
      self.output = input
   end
   return self.output
end

function DropoutConstant:updateNoise()
   self.noise = torch.Tensor()
end

function DropoutConstant:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function DropoutConstant:__tostring__()
   return string.format('%s(%.3f, active=%s)', torch.type(self), self.p, self.active)
end