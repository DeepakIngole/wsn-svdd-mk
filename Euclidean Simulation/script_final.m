addpath(genpath('misc'));
addpath(genpath('lof'));
addpath(genpath('svdd'));
addpath(genpath('solver'));

%% Data Reading
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

%% Sensor Data
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
moteIDs=[1 2 33 35 37];
trainData=[];
for i=moteIDs
    trainData=[trainData;ibrlData(ibrlData(:,1)==i,1:3)];
end

% Visualization
colorList=rand(length(moteIDs),3);
for i=1:length(moteIDs)
    figure(1);
    subplot(211);
    plot(ibrlData(ibrlData(:,1)==i,2),...
         '-','Color',colorList(i,:));
    hold on;
    subplot(212);
    plot(ibrlData(ibrlData(:,1)==i,3),...
         '-','Color',colorList(i,:));
    hold on;
    figure(2);
    plot(trainData(trainData(:,1)==moteIDs(i),2),...
         trainData(trainData(:,1)==moteIDs(i),3),...
         '*','Color',colorList(i,:));
    hold on;
end

save train_data trainData;

%% Data Labelling
clear all;close all;clc;
load train_data;
trainData=consolidator(trainData,[],@mean,2e-2);

[suspicious_index,lof]=local_outlier_factor(trainData(:,2:3),50);

[~,I]=sort(suspicious_index,1,'descend');
negativeIndex=suspicious_index(1:ceil(1e-2*length(I)));
positiveIndex=suspicious_index(ceil(1e-2*length(I))+1:end);

positiveData=trainData(positiveIndex,2:3);
negativeData=trainData(negativeIndex,2:3);

figure(3);clf;
scatter(positiveData(:,1),positiveData(:,2),'bo');
hold on;
scatter(negativeData(:,1),negativeData(:,2),'rx');
hold on;

save labelled_data positiveData negativeData;

%% SVDD Parameters Optimization
clear all;close all;clc;
load labelled_data;

normalizedData=bsxfun(@rdivide,...
    positiveData-repmat(min(positiveData),size(positiveData,1),1),...
    max(positiveData)-min(positiveData));
trainData=consolidator(normalizedData,[],@mean,3e-2);
trainLabel=ones(size(trainData,1),1);

ocSVM.normalizeLB=min(positiveData);
ocSVM.normalizeUB=max(positiveData);

% Derivative-free optimization
algorithmList={NLOPT_GN_DIRECT NLOPT_GN_DIRECT_L NLOPT_GN_DIRECT_L_RAND...
               NLOPT_GN_CRS2_LM...
               NLOPT_GN_ESCH...
               NLOPT_GN_ISRES};
opt.algorithm=algorithmList{3};
opt.xtol_abs=[1e-3;1e-3];
opt.ftol_abs=1e-4;
opt.maxeval=1e2;
opt.max_objective=...
    @(x) svdd_gmean(ocSVM,trainData,trainLabel,positiveData,negativeData,x);
opt.lower_bounds=[1e-3;1e-3];
opt.upper_bounds=[1;1];
opt.verbose=1;
opt.initial_step=[1e-2;1e-2];
xopt=nlopt_optimize_mex(opt,[.5 .5]);

% Optimal hyperparameters
ocSVM.C=[xopt(1) 0];
ocSVM.sigma=xopt(2);
ocSVM=svdd_optimize(ocSVM,trainData,trainLabel);
save ocsvm_result ocSVM;

% Validation
testData=repmat(ocSVM.normalizeLB,1e4,1)+...
    rand(1e4,2).*(ocSVM.normalizeUB-ocSVM.normalizeLB);
predictLabel=svdd_classify(ocSVM,testData);

figure(4);clf;
plot(positiveData(:,1),positiveData(:,2),'r*');hold on;
plot(negativeData(:,1),negativeData(:,2),'b*');hold on;
plot(testData(predictLabel==1,1),testData(predictLabel==1,2),'go','linewidth',2);hold on;
plot(testData(predictLabel==-1,1),testData(predictLabel==-1,2),'ko','linewidth',2);
xlim([ocSVM.normalizeLB(1) ocSVM.normalizeUB(1)]);
ylim([ocSVM.normalizeLB(2) ocSVM.normalizeUB(2)]);



