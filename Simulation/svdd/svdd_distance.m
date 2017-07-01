function sphereDistance=svdd_distance(ocSVM,trainData)

K=kernel_NewData(trainData,ocSVM.supportVector,ocSVM.invS,ocSVM.sigma);
sphereDistance=-2*sum((ones(size(trainData,1),1)*ocSVM.alpha').*K, 2);
