function setParamsValue(obj,params,value)
% INITPARAM  Initialize the paramers of the DagNN
%   OBJ.INITPARAM() uses the INIT() method of each layer to initialize
%   the corresponding parameters (usually randomly).

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
if ~iscell(params)
    params={params};
end
% the terms of the BSD license (see the COPYING file).
if ~iscell(value)
    value={value};
end
p = obj.getParamIndex(params) ;

for i=length(p)
    if size(obj.params(p(i)).value)~=size(value{i})
        error(['Param' params{i} 'has not been initialled or The size of updating value is wrong.'])
    end
end
switch obj.device
    case 'cpu'
        value = cellfun(@gather, value, 'UniformOutput', false) ;
    case 'gpu'
        value = cellfun(@gpuArray, value, 'UniformOutput', false) ;
end


[obj.params(p).value] = deal(value{:}) ;
