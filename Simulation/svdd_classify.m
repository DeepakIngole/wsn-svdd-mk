function predictLabel=svdd_classify(ocSVM,testData)

sigma=ocSVM.sigma;
squaredRadius=ocSVM.squaredRadius;
supportVector=ocSVM.supportVector;
alpha=ocSVM.alpha;

N=size(testData,1);
testData=(testData-repmat(.5*(ocSVM.normalizeLB+ocSVM.normalizeUB),N,1))...
    ./(ocSVM.normalizeUB-ocSVM.normalizeLB);

K=exp(-(bsxfun(@plus,sum(testData.*testData,2),sum(supportVector.*supportVector,2)')...
    -2*testData*supportVector')/(sigma*sigma));
predictLabel=sign(squaredRadius+2*sum(repmat(alpha',N,1).*K,2));
        