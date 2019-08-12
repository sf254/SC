load data487by22
TCL=sorted(:,end);% the last column TCL total crack length (mm)
input=sorted(:,1:end-1);% chmeistry & welding parameters

p0=input';
t0=TCL';
%--------------------------------------------------------------------------
n_samples=size(t0,2);
[p,ps_in] = mapminmax(p0,-0.5,0.5);% ps_in is the transformation matrix 
[t,ts_out] = mapminmax(t0,-0.99,0.99);%ts_out is the transformation matrix
%--------------------------------------------------------------------------
maxium_epochs=10;%Autoencoder training epoch
maxium_epochs_net0=1000;% DNN training epoch

seed=14618;%for reproducibility, set random number seed

kk1=6;%neuron number in 1st hidden layer 
kk2=5;%neuron number in 2nd hidden layer
kk3=4;%neuron number in 3rd hidden layer
kk4=3;%neuron number in 4th hidden layer

%% build 5 autoencoders: net1,net2,net3,net4,net5
net1=newff(p,p,kk1,{'tansig'});
net1.inputs{1}.processFcns={};% cancel the default process function
net1.outputs{2}.processFcns={};% cancel the default process function
net1.trainFcn='trainbr';% train algorithm Bayesian regulation backpro
net1.trainParam.epochs=maxium_epochs;%maxium epochs
net1.divideFcn='divideind';% train/test data dividing 
tr_ind=1:1:n_samples;
te_ind=1:3:n_samples;

tr_ind(te_ind)=[];
[net1.divideParam.trainInd,net1.divideParam.valInd,net1.divideParam.testInd] = ...
                    divideind(n_samples,tr_ind,[],te_ind);
rng(seed);                
net1=init(net1);
[net1,tr]=train(net1,p,p);

p1=tansig(net1.IW{1,1}*p+net1.b{1}*ones(1,n_samples));
%-------------------------------------------------------------------------
net2=newff(p1,p1,kk2,{'tansig'});
net2.inputs{1}.processFcns={};% cancel the default process function
net2.outputs{2}.processFcns={};% cancel the default process function
net2.trainFcn='trainbr';% train algorithm Bayesian regulation backpro
net2.trainParam.epochs=maxium_epochs;%maxium epochs
net2.divideFcn='divideind';% train/test data dividing 
[net2.divideParam.trainInd,net2.divideParam.valInd,net2.divideParam.testInd] = ...
                    divideind(n_samples,tr_ind,[],te_ind);
rng(seed);
net2=init(net2);
[net2,tr2]=train(net2,p1,p1);
p2=tansig(net2.IW{1,1}*p1+net2.b{1}*ones(1,n_samples));
%-------------------------------------------------------------------------
net3=newff(p2,p2,kk3,{'tansig'});
net3.inputs{1}.processFcns={};% cancel the default process function
net3.outputs{2}.processFcns={};% cancel the default process function
net3.trainFcn='trainbr';% train algorithm Bayesian regulation backpro
net3.trainParam.epochs=maxium_epochs;%maxium epochs
net3.divideFcn='divideind';% train/test data dividing 
[net3.divideParam.trainInd,net3.divideParam.valInd,net3.divideParam.testInd] = ...
                    divideind(n_samples,tr_ind,[],te_ind);
 rng(seed);
 net3=init(net3);
[net3,tr3]=train(net3,p2,p2);
p3=tansig(net3.IW{1,1}*p2+net3.b{1}*ones(1,n_samples));
%--------------------------------------------------------------------------
net4=newff(p3,p3,kk4,{'tansig'});
net4.inputs{1}.processFcns={};% cancel the default process function
net4.outputs{2}.processFcns={};% cancel the default process function
net4.trainFcn='trainbr';% train algorithm Bayesian regulation backpro
net4.trainParam.epochs=maxium_epochs;%maxium epochs
net4.divideFcn='divideind';% train/test data dividing 
[net4.divideParam.trainInd,net4.divideParam.valInd,net4.divideParam.testInd] = ...
                    divideind(n_samples,tr_ind,[],te_ind);
rng(seed);     
net4=init(net4);
[net4,tr4]=train(net4,p3,p3);
p4=tansig(net4.IW{1,1}*p3+net4.b{1}*ones(1,n_samples));
%--------------------------------------------------------------------------
net5=newff(p4,p4,1,{'tansig','purelin'});
net5.inputs{1}.processFcns={};% cancel the default process function
net5.outputs{2}.processFcns={};% cancel the default process function
net5.trainFcn='trainbr';% train algorithm Bayesian regulation backpro
net5.trainParam.epochs=maxium_epochs;%maxium epochs
net5.divideFcn='divideind';% train/test data dividing 
[net5.divideParam.trainInd,net5.divideParam.valInd,net5.divideParam.testInd] = ...
                    divideind(n_samples,tr_ind,[],te_ind);
rng(seed);    
net5=init(net5);
[net5,tr5]=train(net5,p4,p4);
%%-------------------------------------------------------------------------
%%train 4 hidden layers DNN using the autoencoder inilization 
net=newff(p,t,[kk1,kk2,kk3,kk4],{'tansig','tansig','tansig','tansig'});
net.inputs{1}.processFcns={};% cancel the default process function
net.outputs{5}.processFcns={};% cancel the default process function 4=3hidden+1
net.trainFcn='trainbr';% train algorithm Bayesian regulation backpropagation 
net.trainParam.epochs=maxium_epochs_net0;%maxium epochs
net.trainParam.max_fail=maxium_epochs_net0;
net.divideFcn='divideind';% train/test data dividing 
[net.divideParam.trainInd,net.divideParam.valInd,net.divideParam.testInd] = ...
                    divideind(n_samples,tr_ind,te_ind,[]);
 %rng(seed);
 %net=init(net);
 net.IW{1,1}=net1.IW{1,1};
 net.b{1}=net1.b{1};
 net.LW{2,1}=net2.IW{1,1};
 net.b{2}=net2.b{1};
 net.LW{3,2}=net3.IW{1,1};
 net.b{3}=net3.b{1};
 net.LW{4,3}=net4.IW{1,1};
 net.b{4}=net4.b{1};
 net.LW{5,4}=net5.IW{1,1};
 net.b{5}=net5.b{1};
 
 [net,tr]=train(net,p,t); 
 [r_test,m_test,b_test] = regression(t(:,tr.valInd),net(p(:,tr.valInd)));
 plotregression(t(:,tr.trainInd),net(p(:,tr.trainInd)),'training',t(:,tr.valInd),net(p(:,tr.valInd)),'testing')           
