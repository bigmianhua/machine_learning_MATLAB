clear all
close all
n=500;
L=zeros(20,4);
c_total=zeros(n,4);
u_total=zeros(5,2,4);
w=[0.2,0.5,0.3];
mu=[0,3,0;0,0,3];
sigma=[1,0;0,1];
observation_class=part1(n,w);
observation=zeros(n,2);
n1=find(observation_class==1);
class1=sum(observation_class==1);
n2=find(observation_class==2);
class2=sum(observation_class==2);
n3=find(observation_class==3);
class3=sum(observation_class==3);
observation(n1,:)=mvnrnd(mu(:,1),sigma,size(n1,2));
observation(n2,:)=mvnrnd(mu(:,2),sigma,size(n2,2));
observation(n3,:)=mvnrnd(mu(:,3),sigma,size(n3,2));
iteration=20;
for K=2:5
    [L(:,K-1),u_total(1:K,:,K-1),c_total(:,K-1)]=kmeans(observation,iteration,K,'point');
%     L(:,K-1)=tmp;
%     c_total(:,K-1)=c;
%     u_total(1:K,:,K-1)=u;
end
for i=1:4
    figure,plot(1:iteration,L(:,i));
    title(['objective function L per iteration t for K=',num2str(i+1)]);
    xlabel('iteration t');
    ylabel('objective function L');
end
c_1=find(c_total(:,2)==1);
c_2=find(c_total(:,2)==2);
c_3=find(c_total(:,2)==3);
figure,plot(observation(c_1,1),observation(c_1,2),'b.',observation(c_2,1),observation(c_2,2),'m.',observation(c_3,1),observation(c_3,2),'g.',u_total(1:3,1,2),u_total(1:3,2,2),'rx','LineWidth',4,'MarkerSize',10);
title('Clusters when K=3');
c_1=find(c_total(:,4)==1);
c_2=find(c_total(:,4)==2);
c_3=find(c_total(:,4)==3);
c_4=find(c_total(:,4)==4);
c_5=find(c_total(:,4)==5);
figure,plot(observation(c_1,1),observation(c_1,2),'b.',observation(c_2,1),observation(c_2,2),'m.',observation(c_3,1),observation(c_3,2),'g.',observation(c_4,1),observation(c_4,2),'y.',observation(c_5,1),observation(c_5,2),'c.',u_total(1:5,1,4),u_total(1:5,2,4),'rx','LineWidth',4,'MarkerSize',10);
title('Clusters when K=5');