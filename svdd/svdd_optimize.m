function ocSVM=svdd_optimize(ocSVM,trainData,trainLabel)

N=size(trainData,1);
invS=inv(cov(trainData));

% Cost matrices
K=mahalanobis_kernel(trainData,invS,ocSVM.sigma);
D=(trainLabel*trainLabel').*K;
f=trainLabel.*diag(D);

% Preprocessing Hessian matrix
i=-30;
while 1
    [~,p]=chol(D+(10^i)*eye(N));
    if p==0
        i=i+5;
        D=D+(10^i)*eye(N);
        break;
    else
        i=i+1;
    end
end
D=.5*(D+D');

% Equality constraint
A=trainLabel';
b=1;

% Lower and upper bounds
lb=zeros(N,1);
ub=lb;
ub(trainLabel==1)=ocSVM.C(1);
ub(trainLabel==-1)=ocSVM.C(2);

% Solve dual optimisation
try % use CPLEX if installed
    alpha=cplexqp(2*D,-f,[],[],A,b,lb,ub); 
catch % otherwise use quadprog (not recommended)
    alpha=quadprog(2*D,-f,[],[],A,b,lb,ub); 
end
alpha=trainLabel.*alpha;

% Squared radius (without offset)
sphereDistance=-2*sum((ones(N,1)*alpha').*K, 2);
squaredRadius=mean(sphereDistance((alpha<ub)&(alpha>1e-8),:));

% Errors and support vectors
supportVector=trainData(alpha>1e-8,:);  
alpha=alpha(alpha>1e-8);

% One-class SVM
ocSVM.supportVector=supportVector;
ocSVM.alpha=alpha;
ocSVM.squaredRadius=squaredRadius;
ocSVM.invS=invS;
