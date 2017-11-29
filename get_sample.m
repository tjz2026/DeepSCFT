function [inputn,inputs,outputn,outputs] = get_sample(varargin)
if nargin == 3
    distr = varargin{1};
    n_sample = varargin{2};
    n_train = varargin{3}
    if strcmp(distr,'random') 
       x1=2*unifrnd(0,1,1,n_sample)-1;
       x2=2*unifrnd(0,1,1,n_sample)-1;
       y= zeros(1,n_sample);
       y= func2(x1,x2);
       input=[x01',x02']
       output=y';
       k=rand(1,n_sample);
       [m,n]=sort(k);
       input_train=input(n(1:n_train),:)';
       output_train=output(n(1:n_train),:)';
       [inputn,inputs]=mapminmax(input_train);
       [outputn,outputs]=mapminmax(output_train);
    end
elseif nargin == 4
    distr = varargin{1};
    nx = varargin{2};
    ny = varargin{3};
    n_sample = (nx+1)*(ny+1);
    n_train = varargin{4};
    if strcmp(distr,'even')
    scale1 = 1.0/(nx/2.0); 
    scale2 = 1.0/(ny/2.0);
    x1 = [-nx/2:nx/2]*scale1;
    x2 = [-ny/2:ny/2]*scale2
    [X1,Y1]=meshgrid(x1,x2);
    x01 = reshape(X1,1,n_sample);
    x02 = reshape(Y1,1,n_sample);
    y= func2(x01,x02);
    input=[x01',x02'];
    output=y';
    k=rand(1,n_sample);
    [m,n]=sort(k);
    input_train=input(n(1:n_train),:)';
    output_train=output(n(1:n_train),:)';
    [inputn,inputs]=mapminmax(input_train);
    [outputn,outputs]=mapminmax(output_train);
    end
end

        

