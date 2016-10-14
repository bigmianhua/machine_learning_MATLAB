function w=logistics(x,label,step)
[m,n]=size(x);
w=zeros(m,1);
for i=1:n
    temp=step*(1-1/(1+exp(-label(i)*x(:,i)'*w)))*label(i)*x(:,i);
    w=w+temp;
end
end