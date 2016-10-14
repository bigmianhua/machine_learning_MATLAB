clear all
close all
load mnist_mat
py=zeros(1,10);
meany=zeros(20,10);
sigmay=zeros(20,20,10);
for dig=0:9
    e=find(label_train==dig);
    py(dig+1)=length(e)/5000;
    x_e=Xtrain(:,e);
    meany(:,dig+1)=sum(x_e,2)./length(e);
    for i=1:length(e)
        x_e_c=x_e(:,i);
        temp=(x_e_c-meany(:,dig+1))*(x_e_c-meany(:,dig+1))';
        sigmay(:,:,dig+1)=sigmay(:,:,dig+1)+temp;
    end
    sigmay(:,:,dig+1)=sigmay(:,:,dig+1)./length(e);
end
C=zeros(10,10);
misindex=zeros(3,3);
misi=1;
for i=1:500
    testdata=Xtest(:,i);
    maxout=0;
    yp=-1;
    for dig=0:9
        testout=mvnpdf(testdata,meany(:,dig+1),sigmay(:,:,dig+1));
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
for dig=1:10
    fdata=Q*meany(:,dig);
    figure,imagesc(fliplr(rot90(reshape(fdata,28,28),3)));
    title(['Mean of y',num2str(dig)]);
end
pb=zeros(10,3);
for i=1:3
    figuredata=Xtest(:,misindex(1,i));
    fdata=Q*figuredata;
    figure,imagesc(fliplr(rot90(reshape(fdata,28,28),3)));
    title(['Misclassified Example ',num2str(i),' Bayes Classifer',' true=',num2str(misindex(2,i)),' prediction=',num2str(misindex(3,i))]);
    for dig=0:9
        pb(dig+1,i)=mvnpdf(Xtest(:, misindex(1,i)),meany(:,dig+1),sigmay(:,:,dig+1));
    end
end