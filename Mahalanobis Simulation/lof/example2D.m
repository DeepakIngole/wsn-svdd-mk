clear;clc;
load test_data;

k = 20;

[suspicious_index,lof]=local_outlier_factor(data, k);

% [~,I]=sort(lof,1,'descend');
% threshold=lof(ceil(.015*length(I)));
% 
% target=data(lof>=threshold,:);
% normal=data(lof<threshold,:);
% 
% figure(1);clf;
% scatter(normal(:,1),normal(:,2),'b');hold on;
% scatter(target(:,1),target(:,2),'rx');

[~,I]=sort(suspicious_index,1,'descend');
bad=suspicious_index(1:ceil(.05*length(I)));
good=suspicious_index(ceil(.05*length(I))+1:end);

figure(2);clf;
scatter(data(bad,1),data(bad,2),'rx');hold on;
scatter(data(good,1),data(good,2),'b');hold on;
