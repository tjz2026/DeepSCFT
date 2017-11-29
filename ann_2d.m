
% init the sample points randomly
%n_sample=1500;n_train = 1000;
%[input,inputs,outputn,outputs] = get_sample('random',n_sample,n_train);
%init the sample points using regular grid points
nx = 40;
ny = 40;
n_train = (nx+1)*(ny+1);
[inputn,inputs,outputn,outputs] = get_sample('even',nx,ny,n_train);
%bp
%net=newff(inputn,outputn,[100],{'logsig','logsig'},'trainlm');
%net=newff(inputn,outputn,[20],{'tansig'},'trainlm');
%net=newff(input_train,output_train,[40,40],{'tansig','purelin'},'trainlm');
net=newff(inputn,outputn,[100],{'tansig'},'trainlm');
% super paramters of neural network
net.trainParam.epochs=2000;
net.trainParam.lr=0.1;
net.trainParam.goal=0.0000004;
%train
net=train(net,inputn,outputn);
figure(1)
% check for validation
nx = 50;
x2 = [-nx:nx]*(1.0/nx);
input2=[x2',x2'];
[X,Y]=meshgrid(x2,x2);
x11 = reshape(X,(2*nx+1)^2,1);
x12 = reshape(Y,(2*nx+1)^2,1);
input2 = [x11,x12]';
inputn_test=mapminmax('apply',input2,inputs);
an = sim(net,inputn_test);
an2 = mapminmax('reverse',an,outputs)
Z = reshape(an2,2*nx+1,2*nx+1);
%mesh(X,Y,Z);
%hold on
Z2 = func2(x11,x12);
Z2 = reshape(Z2,2*nx+1,2*nx+1);
mesh(X,Y,Z);
legend('predict')
hold on
plot3(x11(1:12:end),x12(1:12:end),func2(x11(1:12:end),x12(1:12:end)),'ro');
legend('exact')