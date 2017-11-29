n_sample=500;
x1=2*unifrnd(0,1,1,n_sample)-1;
y= zeros(1,n_sample);
y= func(x1);

%input=[x1',x2',x3'];
input=[x1'];
output=y';
k=rand(1,n_sample);
[m,n]=sort(k);
n_train = 200;
input_train=input(n(1:n_train),:)';
output_train=output(n(1:n_train),:)';
input_test=input(n(n_sample-n_train+1:n_sample),:)';
output_test=output(n(n_sample-n_train+1:n_sample),:)';
%???????
%[inputn,inputs]=mapminmax(input_train);
%[outputn,outputs]=mapminmax(output_train);
%bp??????
%net=newff(inputn,outputn,[100],{'logsig','logsig'},'trainlm');
%net=newff(inputn,outputn,[20],{'tansig'},'trainlm');
net=newff(input_train,output_train,[20],{'tansig'},'trainlm');
%??????
net.trainParam.epochs=2000;
net.trainParam.lr=0.1;
net.trainParam.goal=0.000004;
%train
%net=train(net,inputn,outputn);
net=train(net,input_train,output_train);
%???????
%inputn_test=mapminmax('apply',input_test,inputs);
%bp????????
%an=sim(net,inputn_test);
%an=sim(net,input_test);
%????????
%BPoutput=mapminmax('reverse',an,outputs);
figure(1)
%plot(BPoutput,'r-*')
%legend('predict')
%hold on
%plot(output_test,'g--')
%legend('exact')
x2 = [-50:50]*0.02;
an2 = sim(net,x2);
plot(x2,an2,'b--')
legend('new predict')
hold on
plot(x2,func(x2),'r*')
legend('new exact')