function ocSVM=svdd_optimize(ocSVM,trainData,trainLabel)

ocSVM.C(1)=min(1,max(1/size(trainData,1),ocSVM.C(1)));

% Cost matrices
K=exp(-(bsxfun(@plus,sum(trainData.^2,2),sum(trainData.^2,2)')...
    -2*trainData*trainData')/ocSVM.sigma^2);
D=(trainLabel*trainLabel').*K;
f=trainLabel.*diag(D);

% Preprocessing Hessian matrix
i=-30;
while 1
    [~,p]=chol(D+(10^i)*eye(size(trainData,1)));
    if p==0
        i=i+5;
        D=D+(10^i)*eye(size(trainData,1));
        break;
    else
        i=i+1;
    end
end
D=.5*(D+D');

% Lower and upper bounds
lb=zeros(size(trainData,1),1);
ub=lb;
ub(trainLabel==1)=ocSVM.C(1);
ub(trainLabel==-1)=ocSVM.C(2);

% Solve dual optimisation
try % use CPLEX if installed
    alpha=cplexqp(2*D,-f,[],[],[],[],lb,ub); 
catch % otherwise use quadprog (not recommended)
    options=optimoptions('quadprog',...
        'Algorithm','trust-region-reflective','Display','off');
    alpha=quadprog(2*D,-f,[],[],[],[],lb,ub,[],options); 
end
alpha=trainLabel.*alpha;

% Squared radius (without offset)
sphereDistance=-2*sum((ones(size(trainData,1),1)*alpha').*K, 2);
squaredRadius=mean(sphereDistance((alpha<ub)&(alpha>1e-8),:));

% Errors and support vectors
supportVector=trainData(alpha>1e-8,:);  
alpha=alpha(alpha>1e-8);

% One-class SVM
ocSVM.supportVector=supportVector;
ocSVM.alpha=alpha;
ocSVM.squaredRadius=squaredRadius;

