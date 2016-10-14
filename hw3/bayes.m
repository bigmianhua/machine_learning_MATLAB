function [flag,w0,w]=bayes(x,label)
py=zeros(1,2);
[m,n]=size(x);
x_del=x(2:m,:);
l=[-1,1];
meany=zeros(m-1,2);
tol=1e-8;
x_shift=x_del;
for dig=1:2
    e=find(label==l(dig));
    if (~isempty(e))        
        py(dig)=length(e)/n;
        x_e=x_del(:,e);
        meany(:,dig)=sum(x_e,2)./length(e);      
        for i=1:length(e)
              x_shift(:,e(i))=x_shift(:,e(i))-meany(:,dig);  
        end
    end
end
sigma=zeros(m-1,m-1);
mu=sum(x_shift,2)/n;
for i=1:n
    x_e_c=x_shift(:,i);
    sigma=sigma+x_e_c*x_e_c';
%     sigma=sigma+(x_e_c-mu)*(x_e_c-mu)';
end
sigma=sigma/n;
if ((py(1)~=0) && (py(2)~=0))
    if (det(sigma)<tol)
        flag=1;
        w0=0;
        w=zeros(9,1);
    else
        w0=log(py(2)/py(1))-0.5*(meany(:,1)+meany(:,2))'/sigma*(meany(:,2)-meany(:,1));
        w=sigma\(meany(:,2)-meany(:,1));
        flag=0;
    end
    if (~isfinite(w0))
        c=1;
    end
elseif (py(1)==0)
    w0=1;
    w=zeros(9,1);
elseif (py(2)==0)
    w0=-1;
    w=zeros(9,1);
end

