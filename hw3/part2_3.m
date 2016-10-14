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
[~,w0,w]=bayes(X_train,label_train);
y_predict_test=zeros(1,n_test);
for i=1:n_test
    y_predict_test(i)=sign(w0+X_test(2:m,i)'*w);
end
temp_test=find(label_test~=y_predict_test);
[~,temp_error]=size(temp_test);
test_acc=1-temp_error/n_test;