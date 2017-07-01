function K=mahalanobis_kernel(trainData,invS,sigma)

K=mahalanobis_distance(invS,trainData);
K=exp(-K./(2*sigma^2));
