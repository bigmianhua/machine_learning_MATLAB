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
step=0.1;
w=logistics(X_train,label_train,step);
y_predict_test=zeros(1,n_test);
for i=1:n_test
        tp=exp(X_test(:,i)'*w)/(1+exp(X_test(:,i)'*w));
        if (tp>(1-tp))
            y_predict_test(i)=1;
        else
            y_predict_test(i)=-1;
        end
end
temp_test=find(label_test~=y_predict_test);
[~,temp_error]=size(temp_test);
test_acc=1-temp_error/n_test;