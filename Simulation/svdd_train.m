function ocSVM=svdd_train(ocSVM,C,sigma,trainData,trainLabel)

    [N,~]=size(trainData);
    C(1)=1/(length(find(trainLabel==1))*C(1));
    C(2)=1/(length(find(trainLabel==-1))*C(2));

    % Kernel matrix
    K=exp(-(ones(N,1)*sum(trainData'.*trainData',1)...
        +sum(trainData.*trainData,2)*ones(1,N)...
        -2.*trainData*trainData')/(sigma*sigma));
    
    % Cost matrices
    D=(trainLabel*trainLabel').*K;
    f=trainLabel.*diag(D);

    % Convexification
    i=-30;
    while (pd_check(D+(10.0^i)*eye(N))==0)
        i=i+1;
    end
    i=i+5;
    D=D+(10.0^i)*eye(N);
    
    % Equality constraints
    A=trainLabel';
    b=1.0;
    
    % Lower and upper bounds
    lb=zeros(N,1);
    ub=lb;
    ub(trainLabel==1)=C(1);
    ub(trainLabel==-1)=C(2);
    
    % Solve QP
    alpha=cplexqp(2.0*D,-f,[],[],A,b,lb,ub);
    alpha=trainLabel.*alpha;
    
    % The support vectors and errors
    I=find(abs(alpha)>1e-8);
    
    % Squared radius
    sphereDistance=-2*sum((ones(N,1)*alpha').*K, 2);
    border=I((alpha(I)<ub(I))&(alpha(I)>1e-8));
    if (size(border,1)<1)
        border=I;
    end
    squaredRadius=mean(sphereDistance(border,:));    
    
    % Support vectors and alpha
    alpha(abs(alpha)<1e-8)=0.0;
    supportVector=trainData(I,:);
    alpha=alpha(I);
    
    % One-class SVM
    ocSVM.sigma=sigma;
    ocSVM.squaredRadius=squaredRadius;
    ocSVM.supportVector=supportVector;
    ocSVM.alpha=alpha;

end

function posdef=pd_check(a) % Check positive definite property of a matrix

    n=size(a,1);
    tol=1e-15;
    posdef=0;
    for j=1:n
        if (j>1)
            a(j:n,j)=a(j:n,j)-a(j:n,1:j-1)*a(j,1:j-1)';
        end;
        if (a(j,j)<tol)
             return;
        end;
        a(j:n,j)=a(j:n,j)/sqrt(a(j,j));
    end;
    posdef=1;

end
