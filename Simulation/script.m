addpath('misc');
addpath('svdd');
addpath(genpath('solver'));

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
I{35}=[];
I{37}=unique(max(2.046e4,min(2.36e4,find(moteData{37}(:,1)>37))));
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
load train_data;

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
plot(abnormalData(:,1),abnormalData(:,2),'b*');hold on;
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
clear predictLabel;
for i=[1 2 33 35 37]
    moteData{i}=ibrlData(ibrlData(:,1)==i,[2 3]);
    predictLabel{i}=svdd_classify(ocSVM,moteData{i});
    figure(i);
    plot(moteData{i},'b-');hold on;
    plot(find((predictLabel{i}==-1)),moteData{i}(predictLabel{i}==-1,:),'ro','linewidth',2);
end

%% Parameters Optimisation using Grid Search
clear all;close all;clc;
load train_data;

normalizedData=bsxfun(@rdivide,...
    normalData-repmat(.5*(max(normalData)+min(normalData)),size(normalData,1),1),...
    max(normalData)-min(normalData));
trainData=consolidator(normalizedData,[],@mean,3e-2);
trainLabel=ones(size(trainData,1),1);

ocSVM.normalizeLB=min(normalData);
ocSVM.normalizeUB=max(normalData);

C_grid=[5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1 1];
sigma_grid=[1/128 1/64 1/32 1/16 1/8 1/4 1/2 1 2 4 8];
[Cgrid,sigmagrid]=meshgrid(C_grid,sigma_grid);

for i=1:size(Cgrid,1)
    
    for j=1:size(Cgrid,2)

        % Training
        ocSVM.C=[Cgrid(i,j) 0];
        ocSVM.sigma=sigmagrid(i,j);
        ocSVM=svdd_optimize(ocSVM,trainData,trainLabel);

        % Evaluation
        normalLabel=svdd_classify(ocSVM,normalData);
        Acc_positive=length(find(normalLabel==1))/length(normalLabel)*100;
        abnormalLabel=svdd_classify(ocSVM,abnormalData);
        Acc_negative=length(find(abnormalLabel==-1))/length(abnormalLabel)*100;
        ggrid(i,j)=sqrt(Acc_positive*Acc_negative);
    
    end
end

% G-mean surface
figure(1);
surf(Cgrid,sigmagrid,ggrid);
xlabel('$C$','interpreter','latex');
ylabel('$\sigma$','interpreter','latex');
zlabel('$g$','interpreter','latex');

% Optimal hyperparameters
[~,ind]=max(ggrid(:));
[m,n]=ind2sub(size(ggrid),ind);
ocSVM.C=[Cgrid(m,n) 0];
ocSVM.sigma=sigmagrid(m,n);
ocSVM=svdd_optimize(ocSVM,trainData,trainLabel);
save ocsvm_result ocSVM;

% Validation
testData=repmat(ocSVM.normalizeLB,3e4,1)+...
    rand(3e4,2).*(ocSVM.normalizeUB-ocSVM.normalizeLB);
predictLabel=svdd_classify(ocSVM,testData);

figure(2);
plot(normalData(:,1),normalData(:,2),'r*');hold on;
plot(abnormalData(:,1),abnormalData(:,2),'b*');hold on;
plot(testData(predictLabel==1,1),testData(predictLabel==1,2),'go','linewidth',2);hold on;
plot(testData(predictLabel==-1,1),testData(predictLabel==-1,2),'ko','linewidth',2);

%% Parameters Optimisation using Global Optimisation
clear all;close all;clc;
load train_data;

normalizedData=bsxfun(@rdivide,...
    normalData-repmat(.5*(max(normalData)+min(normalData)),size(normalData,1),1),...
    max(normalData)-min(normalData));
trainData=consolidator(normalizedData,[],@mean,3e-2);
trainLabel=ones(size(trainData,1),1);

ocSVM.normalizeLB=min(normalData);
ocSVM.normalizeUB=max(normalData);

% Derivative-free optimization
algorithmList={NLOPT_GN_DIRECT NLOPT_GN_DIRECT_L NLOPT_GN_DIRECT_L_RAND...
               NLOPT_GN_CRS2_LM...
               NLOPT_GN_ESCH...
               NLOPT_GN_ISRES};
opt.algorithm=algorithmList{4};
opt.xtol_abs=[1e-3;1e-3];
opt.ftol_abs=1e-4;
opt.maxeval=1e2;
opt.max_objective=...
    @(x) gmean_eval(ocSVM,trainData,trainLabel,normalData,abnormalData,x);
opt.lower_bounds=[5e-3;1/128];
opt.upper_bounds=[1;8];
opt.verbose=1;
opt.initial_step=[1e-1;1e-2];
[ropt,Jopt,nJ]=nlopt_optimize_mex(opt,[1 .5]);

% Optimal hyperparameters
ocSVM.C=[ropt(1) 0];
ocSVM.sigma=ropt(2);
ocSVM=svdd_optimize(ocSVM,trainData,trainLabel);

% Validation
testData=repmat(ocSVM.normalizeLB,1e4,1)+...
    rand(1e4,2).*(ocSVM.normalizeUB-ocSVM.normalizeLB);
predictLabel=svdd_classify(ocSVM,testData);

figure(2);
plot(normalData(:,1),normalData(:,2),'r*');hold on;
plot(abnormalData(:,1),abnormalData(:,2),'b*');hold on;
plot(testData(predictLabel==1,1),testData(predictLabel==1,2),'go','linewidth',2);hold on;
plot(testData(predictLabel==-1,1),testData(predictLabel==-1,2),'ko','linewidth',2);
xlim([ocSVM.normalizeLB(1) ocSVM.normalizeUB(1)]);
ylim([ocSVM.normalizeLB(2) ocSVM.normalizeUB(2)]);

