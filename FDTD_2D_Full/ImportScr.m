% suff='1_';
suff='';
data=load([suff,'Ez.txt']);
% data=load('Hx.txt');
set=load([suff,'FieldSaveDat.txt']);

out=zeros(set(1),length(data(1,:)),set(2));

for i=1:set(2)
    j=i-1;
out(:,:,i)=data((set(1)*j)+1:(set(1)*i),:);
end
% 
% % implay internals
% varargin ='out';
% hScopeCfg = iptscopes.IMPlayScopeCfg;
% 
% if nargin > 0
% 
%     % do a squeeze so we allow M-by-N-by-1-by-K arrays
%     varargin{1} = squeeze(out);%squeeze(varargin{1});
%     
%     hScopeCfg.ScopeCLI = hScopeCfg.createScopeCLI(varargin, ...
%         uiservices.cacheFcnArgNames);
%     
% end
% 
% % Create new scope instance.
% obj = uiscopes.new(hScopeCfg);
% 
% if nargout > 0
%     y = obj;
% end