clear all;close all;clc;

load ibrl_data networkData;

% Normalization
normalizedData=(networkData-repmat(min(networkData),size(networkData,1),1))...
    ./(max(networkData)-min(networkData));
plot(normalizedData(:,1),normalizedData(:,2),'bo');hold on;

% Cardinality reduction
trainData=consolidator(normalizedData,[],@mean,3e-2);
plot(trainData(:,1),trainData(:,2),'r*');

%%
a=prdataset(trainData);
a=oc_set(a,'1');
a=target_class(a);

% first show a 2D plot:
figure; 
clf; 
hold on; 
scatterd(a);

%
w = svdd(a,1e-3,.3);
plotc(w,'r--');hold on;

