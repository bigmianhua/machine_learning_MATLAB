close all
clear all
load faces
iteration=200;
K=25;
tol=1e-16;
[m,n]=size(X);
W=random('unif',0,1,m,K);
H=random('unif',0,1,K,n);
L=zeros(iteration,1);

for t=1:iteration
    H=H.*(W'*X)./(W'*W*H+tol);
    W=W.*(X*H')./(W*H*H'+tol);
    L(t)=sqrt(sum(sum((X-(W*H)).^2)));
end
plot(1:iteration,L);
title('Objective Function (Euclidean Distance) on iteration');
xlabel('Iteration t');
ylabel('Objective Function (Euclidean Distance)');
for i=1:10
    currentw=W(:,i);
    figure,imagesc(reshape(W(:,i),32,32));
    title(['Column ',num2str(i),' of W'])
    [~,index]=max(H(i,:));
    figure,imagesc(reshape(X(:,index),32,32));
    title(['Corresponding column of X of column of H with highest weight of W',num2str(i)]);
end