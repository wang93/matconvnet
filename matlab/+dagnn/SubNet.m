classdef SubNet < dagnn.Layer
    %building...
    properties
        inputs={};%input vars' name in subnet
        outputs={};%output vars' name in subnet
        subnet=dagnn.DagNN();
        name='';
    end
    
    methods
        function outputs = forward(obj, inputs,params)
            %`inputs`\`outputs` is a cell array of the type `{Value1,
            %Value2, Value3,  ...}`, corresponding to the vars in subnet & parentnet
            in=cell(1,length(inputs)*2);
            for i=1:length(inputs)
                in{i*2-1}=obj.inputs{i};
                in{i*2}=inputs{i};
            end
            inputs=in;
            obj.subnet.eval(inputs);
            outputs=obj.subnet.getVar(obj.outputs);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
            obj.subnet.eval(inputs,derOutputs);
            v=obj.subnet.getVarIndex(obj.inputs);
            derInputs=obj.subnet.vars(v).der;
            derParams={};
        end
        
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            inVars=obj.inputs;
            outVars=obj.outputs;
            if numel(inVars)~=numel(inputSizes)
                error('The numbers of input vars don''t match! ')
            end
            sizes=cell(1,2*numel(inVars));
            for i=1:numel(inVars)
                sizes(i*2-1)=inVars(i);
                sizes(i*2)=inputSizes(i);
            end
            varsizes = obj.subnet.getVarSizes( sizes);
                v=obj.subnet.getVarIndex(outVars) ;
                outputSizes=varsizes(v);
  
        end
        
        
        
        function obj = SubNet(varargin)
            obj.load(varargin);
%             for i=1:length(obj.subnet.params)
%                 if isempty(obj.subnet.params(i).value)
%                     error('The params'' value of the subnet is empty! You should set or initial the value!')
%                 end
%             end
            if ~iscell(obj.inputs)
                obj.inputs={obj.inputs};
            end
            if ~iscell(obj.outputs)
                obj.outputs={obj.outputs};
            end
        end
    end
end
