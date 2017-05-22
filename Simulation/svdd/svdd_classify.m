function predictLabel=svdd_classify(ocSVM,testData)

% Normalization
testData=bsxfun(@rdivide,...
    testData-repmat(.5*(ocSVM.normalizeUB+ocSVM.normalizeLB),size(testData,1),1),...
    ocSVM.normalizeUB-ocSVM.normalizeLB);

% Distance from test data to the center of sphere (without offset)
K=exp(-(bsxfun(@plus,sum(testData.^2,2),sum(ocSVM.supportVector.^2,2)')...
    -2*testData*ocSVM.supportVector')/ocSVM.sigma^2);
sphereDistance=-2*sum((ones(size(testData,1),1)*ocSVM.alpha').*K, 2);

% Anomaly detection
predictLabel=ones(size(testData,1),1);
predictLabel(sphereDistance-ocSVM.squaredRadius>1e-6)=-1;
