close all
clear all
load nyt_data
iteration=200;
K=25;
tol=1e-16;
m=3012;
n=8447;
X=zeros(m,n);
W=random('unif',0,1,m,K);
H=random('unif',0,1,K,n);
D=zeros(iteration,1);
for i=1:n
    currentxid=Xid{i};
    currentxcnt=Xcnt{i};
    [~,k]=size(currentxid);
    X(currentxid,i)=currentxcnt;
%     for j=1:k
%         X(nowxid(j),i)=nowxcnt(j);
%     end
end
for t=1:iteration
    temp=X./((W*H)+tol);
    temps=sum(W);
    ts=repmat(temps',[1,size(X,2)]);
    H=H.*((W'*temp)./(ts+tol));
    temp=X./((W*H)+tol);
    temps=sum(H,2);
    ts=repmat(temps',[size(X,1),1]);
    W=W.*((temp*H')./(ts+tol));
    now=W*H;
    D(t)=sum(sum((X.*log(1./(now+tol)))+now));
end
plot(1:iteration,D);
title('Objective Function (Divergence) on iteration');
xlabel('Iteration t');
ylabel('Objective Function (Divergence)');
test=W*H;
a=sum(W);
tw=repmat(a,[size(X,1),1]);
th=repmat(a',[1,size(X,2)]);
W=W./tw;
H=H.*th;
check=W*H-test;
for i=1:10
    noww=W(:,i);
    [weight,word_ind]=sort(noww,'descend');
    disp(['==========Column ',num2str(i),'============']);
    for j=1:10
        word=nyt_vocab(word_ind(j));
        word=word{1};
        disp([word,' weight=',num2str(weight(j))]);
    end
end