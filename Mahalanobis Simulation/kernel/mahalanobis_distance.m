function mahalanobisDistance=mahalanobis_distance(invS,X1,X2)

numberCandidate1=size(X1,1);
for i=1:numberCandidate1
    l1(i)=X1(i,:)*invS*X1(i,:)';
end

if ~exist('X2')
    D1=repmat(l1,numberCandidate1,1);
    D2=repmat(l1',1,numberCandidate1);
    K=X1*invS*X1';
    mahalanobisDistance=D1+D2-2*K;
else
    numberCandidate2=size(X2,1);
    for i=1:numberCandidate2
        l2(i)=X2(i,:)*invS*X2(i,:)';
    end
    D1=repmat(l1,numberCandidate2,1);
    D2=repmat(l2',1,numberCandidate1);
    K=X2*invS*X1';
    mahalanobisDistance=D1+D2-2*K;
end
