addpath('misc');
addpath('svdd');

% %% Data Extraction
% clear all;close all;clc;
% 
% % Read file
% fileID=fopen('data.txt');
% formatSpec='2004-%f-%f %f:%f:%f %f %f %f %f %f %f';
% ibrlData=textscan(fileID,formatSpec);
% fclose(fileID);
% 
% % Extract all data
% month=ibrlData{1};
% date=ibrlData{2};
% 
% hour=ibrlData{3};
% minute=ibrlData{4};
% second=ibrlData{5};
% time=hour*3600+minute*60+second;
% 
% moteid=ibrlData{7};
% temperature=ibrlData{8};
% humidity=ibrlData{9};
% 
% save ibrl_data month date time moteid temperature humidity;

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

% Extract mote data (only sensor 1, 2, 33, 35, 37 are considered)
for i=[1 2 33 35 37]
    moteData{i}=ibrlData(ibrlData(:,1)==i,[2 3]);
end

% Clean mote data using Hampel filter
for i=[1 2 33 35 37]
    if ~isempty(moteData{i})
        moteData{i}(:,1)=hampel(1:length(moteData{i}(:,1)),moteData{i}(:,1));
        moteData{i}(:,2)=hampel(1:length(moteData{i}(:,2)),moteData{i}(:,2));
    end
end

% Convert into vector data
normalData=[];
for i=[1 2 33 35 37]
    normalData=[normalData;moteData{i}];
end

save train_data normalData;

figure(1);
for i=[1 2 33 35 37]
    if ~isempty(moteData{i})
        subplot(211);
        plot(moteData{i}(:,1),'-');hold on;
        subplot(212);
        plot(moteData{i}(:,2),'-');hold on;
    end
end

figure(2);
plot(normalData(:,1),normalData(:,2),'*');

%% Abnormal Data
clear all;close all;clc;
load ibrl_data;

ibrlData=[month date time moteid temperature humidity];
ibrlData(:,[1 2 3])=[]; 

for i=[1 2 33 35 37]
    moteData{i}=ibrlData(ibrlData(:,1)==i,[2 3]);
end

I{1}=[1 2411 2417 2501 2504 2509 2520 2521 2534];
I{2}=[1 276 4602 4611 4613 4628 4630];
I{35}=unique(max(1150,min(2100,find(moteData{35}(:,1)>20.6))));
I{35}(1)=[];I{35}(end)=[];
I{37}=unique(max(2.046e4,min(2.36e4,find(moteData{37}(:,1)>26))));
I{37}(1)=[];I{37}(end)=[];

abnormalData=[];
for i=[1 2 33 35 37]
    abnormalData=[abnormalData;moteData{i}(I{i},:)];
    figure(i)
    plot(moteData{i},'b-');hold on;
    plot(I{i},moteData{i}(I{i},:),'ro');hold on;
end

load train_data;
save train_data normalData abnormalData;

figure(100);
plot(normalData(:,1),normalData(:,2),'g*');hold on;
plot(abnormalData(:,1),abnormalData(:,2),'k*');

%% SVDD Optimisation
clear all;close all;clc;
load train_data normalData;

% Normalization
normalizedData=bsxfun(@rdivide,...
    normalData-repmat(.5*(max(normalData)+min(normalData)),size(normalData,1),1),...
    max(normalData)-min(normalData));

% Cardinality reduction
trainData=consolidator(normalizedData,[],@mean,3e-2);
trainLabel=ones(size(trainData,1),1);

% Training
ocSVM.C=[1 0];
ocSVM.sigma=.5;
ocSVM.normalizeLB=min(normalData);
ocSVM.normalizeUB=max(normalData);
ocSVM=svdd_optimize(ocSVM,trainData,trainLabel);

save ocsvm_result ocSVM;

% Validation
testData=repmat(ocSVM.normalizeLB,1e4,1)+...
    rand(1e4,2).*(ocSVM.normalizeUB-ocSVM.normalizeLB);
predictLabel=svdd_classify(ocSVM,testData);

figure(1);
plot(normalData(:,1),normalData(:,2),'r*');hold on;
plot(testData(predictLabel==1,1),testData(predictLabel==1,2),'go','linewidth',2);hold on;
plot(testData(predictLabel==-1,1),testData(predictLabel==-1,2),'ko','linewidth',2);

%% Time Series Validation
clear all;close all;clc;
load ocsvm_result;
load ibrl_data;

% Extract data given time window
ibrlData=[month date time moteid temperature humidity];
% ibrlData(isnan(ibrlData(:,5)),:)=[];
% ibrlData(isnan(ibrlData(:,6)),:)=[];
% ibrlData(ibrlData(:,1)~=3,:)=[];  
% ibrlData(ibrlData(:,2)>15,:)=[];
ibrlData(:,[1 2 3])=[]; 

% Detect anomalies
for i=[1 2 33 35 37]
    moteData{i}=ibrlData(ibrlData(:,1)==i,[2 3]);
    predictLabel{i}=svdd_classify(ocSVM,moteData{i});
    figure(i);
    plot(moteData{i},'b-');hold on;
    plot(find((predictLabel{i}==-1)),moteData{i}(predictLabel{i}==-1,:),'ro','linewidth',2);
end

%% G-mean Accuracy
clear all;close all;clc;
load ocsvm_result;
load train_data;

% Normal data classification
normalLabel=svdd_classify(ocSVM,normalData);
Acc_positive=length(find(normalLabel==1))/length(normalLabel)*100;

% Abnormal data classification
abnormalLabel=svdd_classify(ocSVM,abnormalData);
Acc_negative=length(find(abnormalLabel==-1))/length(abnormalLabel)*100;

% G-mean accuracy
g=sqrt(Acc_positive*Acc_negative);

