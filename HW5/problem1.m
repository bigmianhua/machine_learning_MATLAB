close all
clear all
load CFB2015
iteration=2500;
n=759;
[m,~]=size(scores);
w=zeros(iteration+1,n);
w(1,:)=ones(1,n)/n;
dif=zeros(iteration,1);
M=zeros(n,n);
for i=1:m
    if (scores(i,2)>scores(i,4))
        M(scores(i,1),scores(i,1))=M(scores(i,1),scores(i,1))+1+scores(i,2)/(scores(i,2)+scores(i,4));
        M(scores(i,3),scores(i,1))=M(scores(i,3),scores(i,1))+1+scores(i,2)/(scores(i,2)+scores(i,4));
        M(scores(i,3),scores(i,3))=M(scores(i,3),scores(i,3))+scores(i,4)/(scores(i,2)+scores(i,4));
        M(scores(i,1),scores(i,3))=M(scores(i,1),scores(i,3))+scores(i,4)/(scores(i,2)+scores(i,4));
    else
        M(scores(i,1),scores(i,1))=M(scores(i,1),scores(i,1))+scores(i,2)/(scores(i,2)+scores(i,4));
        M(scores(i,3),scores(i,1))=M(scores(i,3),scores(i,1))+scores(i,2)/(scores(i,2)+scores(i,4));
        M(scores(i,3),scores(i,3))=M(scores(i,3),scores(i,3))+1+scores(i,4)/(scores(i,2)+scores(i,4));
        M(scores(i,1),scores(i,3))=M(scores(i,1),scores(i,3))+1+scores(i,4)/(scores(i,2)+scores(i,4));
    end
end
for i=1:n
    M(i,:)=M(i,:)/sum(M(i,:));
end
% [V,D]=eig(M');
% [eigindex,~]=find(D==max(max(D)));
% w_inf=V(:,eigindex)'/sum(V(:,eigindex));
[V,D]=eigs(M',1);
w_inf=V'/sum(V);
for t=1:iteration
    w(t+1,:)=w(t,:)*M;
    dif(t)=sum(abs(w(t+1,:)-w_inf));
end
index=[10,100,1000,2500];
for j=1:4
    noww=w(index(j)+1,:);
    [ranking,name]=sort(noww,'descend');
    disp(['========iteration ',num2str(index(j)),'========']);
    for k=1:25
        disp(['Rank ',num2str(k),': ',num2str(name(k)),' Value=',num2str(ranking(k))]);
    end
end
figure,plot(1:iteration,dif,'b');
title('||w_t-w_{\infty}||_1 on iteration');
xlabel('iteration');
ylabel('||w_t-w_{\infty}||_1');
disp(['||w_2500-w_inf||_1=',num2str(dif(2500))]);