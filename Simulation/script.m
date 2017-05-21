%% Data Extraction
clear all;close all;clc;

% Read file
fileID=fopen('data.txt');
formatSpec='2004-%f-%f %f:%f:%f %f %f %f %f %f %f';
ibrlData=textscan(fileID,formatSpec);
fclose(fileID);

% Extract all data
month=ibrlData{1};
date=ibrlData{2};

hour=ibrlData{3};
minute=ibrlData{4};
second=ibrlData{5};
time=hour*3600+minute*60+second;

moteid=ibrlData{7};
temperature=ibrlData{8};
humidity=ibrlData{9};

save ibrl_data month date time moteid temperature humidity;
%% Normal Data
clear all;close all;clc;
load ibrl_data;

% Extract data given time window
ibrlData=[month date time moteid temperature humidity];
ibrlData(isnan(ibrlData(:,5)),:)=[];
ibrlData(isnan(ibrlData(:,6)),:)=[];
ibrlData(ibrlData(:,1)~=3,:)=[];  
ibrlData(ibrlData(:,2)>10,:)=[];
ibrlData(:,[1 2 3])=[]; 

% Extract mote data
for i=[1:54]
    moteData{i}=ibrlData(ibrlData(:,1)==i,[2 3]);
end

% Clean data manually
moteData{5}=[];
moteData{15}=[];
moteData{18}([15e3:end],:)=[];

% Clean mote data using Hampel filter
for i=[1:54]
    if ~isempty(moteData{i})
        moteData{i}(:,1)=hampel(1:length(moteData{i}(:,1)),moteData{i}(:,1));
        moteData{i}(:,2)=hampel(1:length(moteData{i}(:,2)),moteData{i}(:,2));
    end
end

% Convert into vector data
normalData=[];
for i=[1:54]
    normalData=[normalData;moteData{i}];
end

save normal_data normalData;

figure;
for i=[1:54]
    if ~isempty(moteData{i})
        subplot(211);
        plot(moteData{i}(:,1),'-');hold on;
        subplot(212);
        plot(moteData{i}(:,2),'-');hold on;
    end
end

figure;
plot(normalData(:,1),normalData(:,2),'*');

%% Train SVDD
clear all;close all;clc;

load normal_data normalData;

% Normalization
normalizedData=(normalData-repmat(.5*(max(normalData)+min(normalData)),size(normalData,1),1))...
    ./(max(normalData)-min(normalData));

% Cardinality reduction
trainData=consolidator(normalizedData,[],@mean,3e-2);
trainLabel=ones(size(trainData,1),1);

figure(1);
plot(normalData(:,1),normalData(:,2),'r*');hold on;

%
ocSVM.normalizeLB=min(normalData);
ocSVM.normalizeUB=max(normalData);
ocSVM=svdd_optimize(ocSVM,[1/437 0],.3,trainData,trainLabel);

testData=ocSVM.normalizeLB+rand(1e3,2).*(ocSVM.normalizeUB-ocSVM.normalizeLB);
predictLabel=svdd_classify(ocSVM,testData);

figure(1);
plot(testData(predictLabel==1,1),testData(predictLabel==1,2),'go','linewidth',2);hold on;
plot(testData(predictLabel==-1,1),testData(predictLabel==-1,2),'ko','linewidth',2);hold on;

save ocsvm_result ocSVM;





