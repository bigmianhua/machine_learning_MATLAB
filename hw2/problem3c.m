clear all
close all
load mnist_mat
C=zeros(10,10);
w=zeros(21,10);
dw=zeros(21,10);
L=zeros(1,1000);
p=0.1/5000;
newXtrain=ones(21,5000);
newXtrain(1:20,:)=Xtrain;
newXtest=ones(21,500);
newXtest(1:20,:)=Xtest;
misindex=zeros(3,3);
misi=1;
for itr=1:1000
    for i=1:5000
        traindata=newXtrain(:,i);
        label=label_train(i);
        temp=exp(traindata'*w);
        L(itr)=L(itr)+traindata'*w(:,label+1)-log(sum(temp));
    end
    dw=zeros(21,10);
    for dig=0:9
        e=find(label_train==dig);
        traindataset=newXtrain(:,e);
        sumx=sum(traindataset,2);
        temp=zeros(21,1);
        for i=1:5000
            traindata=newXtrain(:,i);
            temp_mother=sum(exp(traindata'*w));
            temp_son=traindata*exp(traindata'*w(:,dig+1));
            temp=temp+temp_son/temp_mother;
        end
        dw(:,dig+1)=sumx-temp;
    end
    w=w+p*dw;
end
for i=1:500
    testdata=newXtest(:,i);
    maxout=0;
    yp=-1;
    for dig=0:9
        temp=exp(testdata'*w);
        testout=exp(testdata'*w(:,dig+1))/sum(temp);
        if (yp==-1)
            maxout=testout;
            yp=dig;
        elseif (testout>maxout)
            maxout=testout;
            yp=dig;
        end
    end
    C(label_test(i)+1,yp+1)=C(label_test(i)+1,yp+1)+1;
    if ((label_test(i)~=yp) && (misi<4))
       misindex(1,misi)=i;
       misindex(2,misi)=label_test(i);
       misindex(3,misi)=yp;
       misi=misi+1;
    end
end
predict_acc=trace(C)/500;
pb=zeros(3,10);
for i=1:3
    figuredata=Xtest(:,misindex(1,i));
    fdata=Q*figuredata;
    figure,imagesc(fliplr(rot90(reshape(fdata,28,28),3)));
    title(['Misclassified Example ',num2str(i),' Multiclass Logistic Classifer',' true=',num2str(misindex(2,i)),' prediction=',num2str(misindex(3,i))]);
    for dig=0:9
        pb(i,dig+1)=exp(newXtest(:,misindex(1,i))'*w(:,dig+1))/sum(exp(newXtest(:,misindex(1,i))'*w));
    end
end
x=1:1000;
figure,plot(x,L);
title('L of iteration');