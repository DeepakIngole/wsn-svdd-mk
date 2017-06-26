function K=kernel_NewData(testData,trainData,invS,sigma)

K=mahalanobis_distance(invS,trainData,testData);
K=exp(-K./(2*sigma^2));
