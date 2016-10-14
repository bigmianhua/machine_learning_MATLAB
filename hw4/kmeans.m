function [L,u,c]=kmeans(observation,iteration,K,initial_method)
[n,m]=size(observation);
initial=zeros(1,K);
i=1;
L=zeros(iteration,1);
if (strcmp(initial_method,'point'))  %initial centeroid with random selected point
    while (i<=K)
        initial(i)=random('unid',n,[1,1]);
        if (sum(initial==initial(i))==1)
            i=i+1;
        end
    end
    u=observation(initial,:);
elseif (strcmp(initial_method,'random'))   %initial centeroid with random vector
    u=zeros(K,m);
    for k = 1:K
        u(k,:) = rand(1,m);
    end
end
c=zeros(n,1);
temp=zeros(n,K);
for t=1:iteration       
    for i=1:K
        temp(:,i)=sum((observation-repmat(u(i,:),[size(observation,1),1])).^2,2);
    end
    [tp,c]=min(temp,[],2);
    %         L(t,K-1)=sum(tp);
    for i=1:K
        num=find(c==i);
        [nk,~]=size(num);
        u(i,:)=sum(observation(num,:),1)/nk;
    end
    for i=1:K
        num=find(c==i);
        [nk,~]=size(num);
        tmp=sum(sum((observation(num,:)-repmat(u(i,:),[nk,1])).^2,2));
        L(t)=L(t)+tmp;
    end
end
end