clear all
close all
load movie_ratings
[~,D1]=size(user);
[~,D2]=size(movie);
[test_num,~]=size(ratings_test);
d=10;
lambda=10;
sigma=0.25;
u=zeros(D1,d);
mu=zeros(1,d);
tmp=ones(1,d)/lambda;
D=diag(tmp);
v=mvnrnd(mu,D,D2)';
iteration=100;
RMSE_train=zeros(1,iteration);
RMSE=zeros(1,iteration);
prediction=zeros(test_num,1);
loglikehoodu=zeros(1,D1);
loglikehoodv=zeros(1,D2);
loglikehood=zeros(1,iteration);
for t=1:iteration
    for i=1:D1
        ratedid=user(i).movie_id;
        ratedscore=user(i).rating;
        tmp=ones(1,d)*lambda*sigma;
        left=diag(tmp);
        right=zeros(d,1);
        [~,num]=size(ratedid);
        for k=1:num
            left=left+v(:,ratedid(k))*v(:,ratedid(k))';
            right=right+v(:,ratedid(k))*ratedscore(k);
        end
%         ratedobject=v(:,ratedid);        
%         for k=1:size(ratedobject,2)
%             left=left+ratedobject(:,k)*ratedobject(:,k)';
%             right=right+ratedobject(:,k)*ratedscore(k);
%         end
        u(i,:)=left\right;
        loglikehoodu(i)=log(mvnpdf(u(i,:),zeros(1,d),D));
    end
    for j=1:D2
        ratedid=movie(j).user_id;
        ratedscore=movie(j).rating;
        tmp=ones(1,d)*lambda*sigma;
        left=diag(tmp);
        right=zeros(d,1);
        [~,num]=size(ratedid);
        for k=1:num
            left=left+u(ratedid(k),:)'*u(ratedid(k),:);
            right=right+u(ratedid(k),:)'*ratedscore(k);
        end
%         ratedobject=u(ratedid,:);        
%         for k=1:size(ratedobject,1)
%             left=left+ratedobject(k,:)'*ratedobject(k,:);
%             right=right+ratedobject(k,:)'*ratedscore(k);
%         end
        v(:,j)=left\right;
        loglikehoodv(j)=log(mvnpdf(v(:,j),zeros(d,1),D));
    end
    total=0;
    loglikehoodm=0;
    for i=1:D1
        ratedid=user(i).movie_id;
        ratedscore=user(i).rating;
        [~,num]=size(ratedid);
        for k=1:num
            total=total+1;
            temp=round(u(i,:)*v(:,ratedid(k)));
            if (temp<1)
                temp=1;
            end
            if (temp>5)
                temp=5;
            end
            loglikehoodm=loglikehoodm+log(mvnpdf(ratedscore(k),u(i,:)*v(:,ratedid(k)),sigma));
            RMSE_train(t)=RMSE_train(t)+(temp-ratedscore(k)).^2;
        end        
    end
    RMSE_train(t)=sqrt(RMSE_train(t)/total);
    loglikehood(t)=loglikehoodm+sum(loglikehoodu)+sum(loglikehoodv);
    for k=1:test_num
        prediction(k)=round(u(ratings_test(k,1),:)*v(:,ratings_test(k,2)));
        if (prediction(k)<1)
            prediction(k)=1;
        end
        if (prediction(k)>5)
            prediction(k)=5;
        end
    end
    RMSE(t)=sqrt(sum((prediction-ratings_test(:,3)).^2)/test_num);
end
figure,plot(2:iteration,RMSE(2:iteration));
xlabel('Iteration t');
ylabel('RMSE');
title('RMSE for test set in 100 iteration');
figure,plot(2:iteration,RMSE_train(2:iteration));
xlabel('Iteration t');
ylabel('RMSE');
title('RMSE for training set in 100 iteration');
figure,plot(2:iteration,loglikehood(2:iteration));
xlabel('Iteration t');
ylabel('Log Joint Likelihood');
title('Log Joint Likelihood in 100 iteration');
disp('=======Problem 3=======');
querymovie=[50,225,300];
order_str=[{'1st'},{'2nd'},{'3rd'},{'4th'},{'5th'},{'6th'},{'7th'},{'8th'},{'9th'},{'10th'}];
for i=1:3
    movieid=querymovie(i);
    nowv=v(:,movieid);
    distance=sqrt(sum((v-repmat(nowv,[1,size(v,2)])).^2,1));
    [B,id]=sort(distance);
    naming=movie_names(movieid);
    disp(['query movie: ',naming{1}]);
    for j=1:5
        naming=movie_names(id(j+1));
        order_num=order_str(j);
        disp([order_num{1},' nearest movie: ',naming{1},', distance=',num2str(B(j+1))]);
    end
end
K=20;
iteration_c=20;
disp('=======initialize with random vector=======');
[u_L,u_centroid,u_cluster]=kmeans(u,iteration_c,K,'random');
num_data=zeros(1,K);
for centroid_id=1:K
    num_data(centroid_id)=sum(u_cluster==centroid_id);
end
[u_d,u_c_id]=sort(num_data,'descend');
for i=1:5
    disp(['Centroid of Cluster ',num2str(i),' on u:']);
    disp(num2str(u_centroid(u_c_id(i),:)));
    dot_u=u_centroid(u_c_id(i),:)*v;
    [dot_u_d,dot_u_id]=sort(dot_u,'descend');
    disp(['Number of Data in Cluster',num2str(i),'=',num2str(u_d(i))]);
    for j=1:10
        naming=movie_names(dot_u_id(j));
        order_num=order_str(j);
        disp([order_num{1},' nearest movie: ',naming{1},', dot product=',num2str(dot_u_d(j))]);
    end
end
disp('=======initialize with random selected datapoint=======');
[u_L,u_centroid,u_cluster]=kmeans(u,iteration_c,K,'point');
num_data=zeros(1,K);
for centroid_id=1:K
    num_data(centroid_id)=sum(u_cluster==centroid_id);
end
[u_d,u_c_id]=sort(num_data,'descend');
for i=1:5
    disp(['Centroid of Cluster ',num2str(i),' on u:']);
    disp(num2str(u_centroid(u_c_id(i),:)));
    dot_u=u_centroid(u_c_id(i),:)*v;
    [dot_u_d,dot_u_id]=sort(dot_u,'descend');
    disp(['Number of Data in Cluster',num2str(i),'=',num2str(u_d(i))]);
    for j=1:10
        naming=movie_names(dot_u_id(j));
        order_num=order_str(j);
        disp([order_num{1},' nearest movie: ',naming{1},', dot product=',num2str(dot_u_d(j))]);
    end
end
disp('=======initialize with random vector=======');
[v_L,v_centroid,v_cluster]=kmeans(v',iteration_c,K,'random');
for centroid_id=1:K
    num_data(centroid_id)=sum(v_cluster==centroid_id);
end
[v_d,v_c_id]=sort(num_data,'descend');
for i=1:5
    disp(['Centroid of Cluster ',num2str(i),' on v:']);
    disp(num2str(v_centroid(v_c_id(i),:)));
    dot_v=sqrt(sum((v-repmat(v_centroid(v_c_id(i),:)',[1,size(v,2)])).^2,1));
    [dot_v_d,dot_v_id]=sort(dot_v);
    disp(['Number of Data in Cluster',num2str(i),'=',num2str(v_d(i))]);
    for j=1:10
        naming=movie_names(dot_v_id(j));
        order_num=order_str(j);
        disp([order_num{1},' nearest movie: ',naming{1},', distance=',num2str(dot_v_d(j))]);
    end
end
disp('=======initialize with random selected datapoint=======');
[v_L,v_centroid,v_cluster]=kmeans(v',iteration_c,K,'point');
for centroid_id=1:K
    num_data(centroid_id)=sum(v_cluster==centroid_id);
end
[v_d,v_c_id]=sort(num_data,'descend');
for i=1:5
    disp(['Centroid of Cluster ',num2str(i),' on v:']);
    disp(num2str(v_centroid(v_c_id(i),:)));
    dot_v=sqrt(sum((v-repmat(v_centroid(v_c_id(i),:)',[1,size(v,2)])).^2,1));
    [dot_v_d,dot_v_id]=sort(dot_v);
    disp(['Number of Data in Cluster',num2str(i),'=',num2str(v_d(i))]);
    for j=1:10
        naming=movie_names(dot_v_id(j));
        order_num=order_str(j);
        disp([order_num{1},' nearest movie: ',naming{1},', distance=',num2str(dot_v_d(j))]);
    end
end