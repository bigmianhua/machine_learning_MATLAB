clear all
close all
load cancer
T=1000;
[m,n]=size(X);
X_test=X(:,1:183);
X_train=X(:,184:683);
label_test=label(:,1:183);
label_train=label(:,184:683);
[~,n_train]=size(X_train);
[~,n_test]=size(X_test);
p=zeros(T+1,n_train);
p(1,:)=1/n_train;
train_error=zeros(1,T);
test_error=zeros(1,T);
a=zeros(1,T);
e=zeros(1,T);
w=zeros(m,T);
step=0.1;
y_predict_train=zeros(1,n_train);
y_predict_test=zeros(1,n_test);
y_predict_train_cu=zeros(1,n_train);
y_predict_test_cu=zeros(1,n_test);
for t=1:T
    B=part1(n_train,p(t,:));
    B_train=X_train(:,B);
    B_label=label_train(B);
    w(:,t)=logistics(B_train,B_label,step);
    for i=1:n_train
        tp=exp(X_train(:,i)'*w(:,t))/(1+exp(X_train(:,i)'*w(:,t)));
        if (tp>(1-tp))
            y_predict_train(i)=1;
        else
            y_predict_train(i)=-1;
        end
    end
    for i=1:n_test
        tp=exp(X_test(:,i)'*w(:,t))/(1+exp(X_test(:,i)'*w(:,t)));
        if (tp>(1-tp))
            y_predict_test(i)=1;
        else
            y_predict_test(i)=-1;
        end
    end
    temp=label_train~=y_predict_train;
    e(t)=sum(p(t,temp));
    a(t)=0.5*log((1-e(t))/e(t));
    pt=p(t,:).*exp(-a(t)*label_train.*y_predict_train);
    p(t+1,:)=pt/sum(pt);
    y_predict_train_cu=y_predict_train_cu+a(t)*y_predict_train;
    y_predict_test_cu=y_predict_test_cu+a(t)*y_predict_test;
    y_predict_train=sign(y_predict_train_cu);
    y_predict_test=sign(y_predict_test_cu);
    temp_test=find(label_test~=y_predict_test);
    [~,temp_error]=size(temp_test);
    test_error(t)=temp_error/n_test;
    temp=find(label_train~=y_predict_train);
    [~,temp_error]=size(temp);
    train_error(t)=temp_error/n_train;
end
figure,plot(1:T,train_error,1:T,test_error);
title('Train Error & Test Error');
legend('Train Error','Test Error');
xlabel('Iteration t');
ylabel('Error rate');
figure,plot(1:T,e);
title('e(t)');
xlabel('Iteration t');
ylabel('e');
figure,plot(1:T,a);
title('a(t)');
xlabel('Iteration t');
ylabel('a');