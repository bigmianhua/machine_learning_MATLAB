function c=part1(n,w)
c=zeros(1,n);
[~,k]=size(w);
cdfw=zeros(1,k);
cdfw(1)=w(1);
for i=2:k
    cdfw(i)=cdfw(i-1)+w(i);
end
temp=random('unif',0,1,[1,n]);
for i=1:n
    for j=1:k
        if (temp(i)<=cdfw(j))
            c(i)=j;
            break;
        end
    end
end