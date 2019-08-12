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
maxium_epochs_net0=1000;% training epoch
seed=14618;%for reproducibility, set random number seed
kk1=21;%neuron number in 1st hidden layer 
%%-------------------------------------------------------------------------
%%train 1 hidden layer SNN using random inilization 
net=newff(p,t,kk1,{'tansig'});
net.inputs{1}.processFcns={};% cancel the default process function
net.outputs{2}.processFcns={};% cancel the default process function 4=3hidden+1
net.trainFcn='trainbr';% train algorithm Bayesian regulation backpropagation 
net.trainParam.epochs=maxium_epochs_net0;%maxium epochs
net.trainParam.max_fail=maxium_epochs_net0;
net.divideFcn='divideind';% train/test data dividing 
[net.divideParam.trainInd,net.divideParam.valInd,net.divideParam.testInd] = ...
                    divideind(n_samples,tr_ind,te_ind,[]);
rng(seed);
net=init(net);
[net,tr]=train(net,p,t); 
[r_test,m_test,b_test] = regression(t(:,tr.valInd),net(p(:,tr.valInd)));
plotregression(t(:,tr.trainInd),net(p(:,tr.trainInd)),'training',t(:,tr.valInd),net(p(:,tr.valInd)),'testing')           
