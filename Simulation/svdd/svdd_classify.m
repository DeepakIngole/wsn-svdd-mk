function [predictLabel,boundaryLabel]=svdd_classify(ocSVM,testData)

% Normalization
testData=bsxfun(@rdivide,...
    testData-repmat(ocSVM.normalizeLB,size(testData,1),1),...
    ocSVM.normalizeUB-ocSVM.normalizeLB);

% Distance from test data to the center of sphere (without offset)
K=kernel_NewData(testData,ocSVM.supportVector,ocSVM.invS,ocSVM.sigma);
sphereDistance=-2*sum((ones(size(testData,1),1)*ocSVM.alpha').*K, 2);

% Anomaly detection
predictLabel=ones(size(testData,1),1);
predictLabel(sphereDistance-ocSVM.squaredRadius>ocSVM.gamma)=-1;

% Boundary decision
boundaryLabel=zeros(size(testData,1),1);
boundaryLabel(sphereDistance-ocSVM.squaredRadius>ocSVM.gamma &...
    sphereDistance-ocSVM.squaredRadius<ocSVM.gamma+1e-3)=1;
