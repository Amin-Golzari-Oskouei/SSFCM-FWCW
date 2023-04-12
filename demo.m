%This demo shows how to call the SSFCM-FWCW algorithm described in the paper:
%M.Hashemzadeh, A.Golzari oskouei and N.farajzadeh, "SSFCM-FWCW: Semi-Supervised Fuzzy C-means
%Method based on Feature-weight and Cluster-weight Learning", Applied Soft Computing, 2023.
%For the demonstration, the iris dataset of the above paper is used.
%Courtesy of A.Golzari Oskouei

clc
clear all
close all

%Load the dataset. The last column of dataset is true labels.
X=load('iris.mat');
X=X.iris;

%delete last column (true labels) in clustering process
class=X(:,end);
X(:,end)=[];

%Normalize data between 0 and 1 (optinal)
[N,d]=size(X);
X=(X(:,:)-min(X(:)))./(max(X(:)-min(X(:))));

%convert class to one-hot encoding
f = double(class == 1:max(class));
f(:,sum(f)==0)=[];

%---------------------
%Algorithm parameters.
%---------------------

%Clustering parameters
k=size(unique(class),1);          %number of clusters.
q=2;                              %the value for the feature weight updates.
p_init=0;                         %initial p.
p_max=0.5;                        %maximum p.
p_step=0.01;                      %p step.
t_max=100;                        %maximum number of iterations.
beta_memory=0;                    %amount of memory for the cluster weight updates.
Restarts=10;                      %number of algorithm restarts.
fuzzy_degree=2;                   %fuzzy membership degree
I=1;                              %The value of this parameter is in the range of (0 and 1]
landa=I./var(X);
landa(landa==inf)=1;

%Semi-supervised parameters
alpha = 2;
labeled_rate = 20;                %rate of labeled data (0-100)

%---------------------
%Cluster the instances using the propsed procedure.
%---------------------

for repeat=1:Restarts
    fprintf('========================================================\n')
    fprintf('proposed clustering algorithm: Restart %d\n',repeat);
    
    % label indicator vector
    rand('state',repeat)
    b = zeros(N,1);
    tmp1=randperm(N);
    b(tmp1(1:N*labeled_rate/100))=1;
    
    %initialize with labeled data.
    if labeled_rate==0
        %Randomly initialize the cluster centers.
        tmp2=randperm(N);
        M=X(tmp2(1:k),:);
    else
        M = ((b.*f)'*X)./repmat(sum(b.*f)',1,d);
    end
    
    %Execute proposed clustering algorithm.
    %Get the cluster assignments, the cluster centers and the cluster weight and feature weight.
    [Cluster_elem,M,W,Z]=SSFCM_FWCW(X,M,k,p_init,p_max,p_step,t_max,beta_memory,N,fuzzy_degree,d,q,landa,f,b, alpha);
    
    %Hard clusters. Select the largest value for each sample among the clusters, and assign that sample to that cluster.
    [~,unsupervised_Cluster]=max(Cluster_elem,[],1);
    
    %Evaluation metrics
    EVAL = Evaluate(class,unsupervised_Cluster');
    Accurcy_unsupervised(repeat)=EVAL(1);
    
    fprintf('End of Restart %d\n',repeat);
    fprintf('========================================================\n\n')
end

fprintf('Average unsupervised accurcy over %d restarts: %f.\n',Restarts,mean(Accurcy_unsupervised));
