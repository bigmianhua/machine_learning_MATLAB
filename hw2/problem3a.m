clear all
close all
load mnist_mat
k=5;
C=zeros(10,10);
distance=zeros(1,5000);
minindex=zeros(1,k);
voter=zeros(1,k);
misindex=zeros(3,3);
misi=1;
for i=1:500
    testdata=Xtest(:,i);
    for j=1:5000
        data=Xtrain(:,j);
        % Euclidean Distance
        distance(j)=sqrt(sum((testdata-data).^2));
    end
    for l=1:k
        [~,minindex(l)]=min(distance);
        distance(minindex(l))=[];
    end
    voter=label_train(minindex);
    v=0;
    yp=0;
    for l=1:k
        for dig=0:9
            e=find(voter==dig);
            if (v<length(e))
                v=length(e);
                yp=dig;
            end
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
for i=1:3
    figuredata=Xtest(:,misindex(1,i));
    fdata=Q*figuredata;
    figure,imagesc(fliplr(rot90(reshape(fdata,28,28),3)));
    title(['Misclassified Example ',num2str(i),' k=',num2str(k),' true=',num2str(misindex(2,i)),' prediction=',num2str(misindex(3,i))]);
end