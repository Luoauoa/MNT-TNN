function [X,iter] = TC_FTNN(M,omega,opts,M_true)

% Solve the Low-Rank Tensor Completion (LRTC) based on Tensor Nuclear Norm (TNN) problem by ADMM
% Reference:
% T.-X. Jiang, M. K. Ng, X.-L. Zhao and T.-Z. Huang, 
% "Framelet Representation of Tensor Nuclear Norm for Third-Order Tensor Completion",
% IEEE Transactions on Image Processing, vol. 29, pp. 7233-7244, 2020.

% Input£º
%       M      - The observed n1*n2*n3 tensor
%       omega  - The observed index set 
%       opts   - The parameters
%       M_true - The original n1*n2*n3 tensor
% Output:
%       X      - The recovered n1*n2*n3 tensor
%       iter   - The iteration number

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');        tol         = opts.tol;         end
if isfield(opts, 'max_iter');   max_iter    = opts.max_iter;	end
if isfield(opts, 'rho');        rho        	= opts.rho;      	end
if isfield(opts, 'lambda1');    lambda1   	= opts.lambda1;   	end
if isfield(opts, 'lambda2');    lambda2   	= opts.lambda2;   	end
if isfield(opts, 'beta');       beta       	= opts.beta;        end
if isfield(opts, 'max_beta');   max_beta    = opts.max_beta;    end
if isfield(opts, 'DEBUG');      DEBUG       = opts.DEBUG;     	end
if isfield(opts, 'Frame');      Frame      	= opts.Frame;     	end
if isfield(opts, 'Level');      Level       = opts.Level;     	end
if isfield(opts, 'wLevel');     wLevel      = opts.wLevel;     	end

%%% Generate Framelet Filters (B_spline)
[D,R]       = GenerateFrameletFilter(Frame);
dim         = size(M);
X           = M;%rand(dim);%ones(dim)*sum(M(omega))/length(omega);
X(omega)    = M(omega);
X_W         = Fold( FraDecMultiLevel(Unfold(X,size(X),3),D,Level) ,  [dim(1:2),dim(3)*size(D,1)*Level],3);

V           = X_W;
Multiplier  = zeros(size(X_W));
nfilter     = 1;
nD          = size(D,1);
muLevel     = ones(size(X_W));
len         = dim(3);
obj_Ori     = 0;
if wLevel<=0
    for ki=1:Level
        for ii=1:nD
            muLevel(:,:,len*(ki-1)*nD+(ii-1)*len+1: len*(ki-1)*nD+ii*len)= muLevel(:,:,len*(ki-1)*nD+(ii-1)*len+1: len*(ki-1)*nD+ii*len)*nfilter*norm(D(ii,:));
        end
        nfilter = nfilter*norm(D(ii,:));
    end
else
    for ki=1:Level
        for ii=1:nD-1
            muLevel(:,:,len*(ki-1)*nD+(ii-1)*len+1: len*(ki-1)*nD+ii*len)=muLevel(:,:,len*(ki-1)*nD+(ii-1)*len+1: len*(ki-1)*nD+ii*len)*nfilter;
        end
        nfilter = nfilter*wLevel;
    end
end

iter = 0;

for iter    = 1 : max_iter
    Vk     = V;Xk = X;

    % update V1
    Nu1     = X_W+Multiplier/beta;
    nn = zeros(size(X_W,3),1);
    for k = 1:size(X_W,3)%indexLF%
        [V(:,:,k),nn(k)]  = prox_nuclear(Nu1(:,:,k),muLevel(1,1,k)*lambda1/beta);%Pro2WNNM
    end
        
    if DEBUG
    obj_V1          = sum(nn)*lambda1;
    end
    % update X
    Nu2     = V - Multiplier/beta ;
    X       = Fold( FraRecMultiLevel(Unfold(Nu2,size(Nu2),3),R,Level), size(X),3);
    X(omega)= M(omega);
    X(X<0)=0;
    X_W     = Fold( FraDecMultiLevel(Unfold(X,size(X),3),D,Level) ,  [dim(1:2),dim(3)*size(D,1)*Level],3);
    obj_Ori = 0;
    if DEBUG>1
        if iter == 1 || mod(iter, 5) == 0
            figure(3);
            for ii = 1:Level
                temp = X_W(:,:,(ii-1)*size(X,3)*size(D,1)+size(X,3)+1:ii*size(X,3)*size(D,1));
                subplot(1,Level,ii);hist(temp(:),1000);title([num2str(ii) '_HF part of XF']);drawnow;
            end
        end
    end
    
    if DEBUG
        for k = 1:size(X_W,3)
            [~,s,~] = mySVD(X_W(:,:,k));
            obj_Ori = obj_Ori+sum(diag(s));
        end
        obj_Ori = obj_Ori+ sum(abs(X_W(:)));
    end
    % update multipliers
    Multiplier = Multiplier + beta*(X_W-V);
    
    chgV1   = max(abs(Vk(:)-V(:)));
    OUT.chgV1(iter) = chgV1;
    chgX    = max(abs(Xk(:)-X(:)));
    OUT.chgX(iter) = chgX;
    chg     = max([chgX chgV1]);
    OUT.chg(iter) = chg;
    if DEBUG
        obj_D   = sum((X_W(:)-V(:)-Multiplier(:)/beta).^2);%+sum((X_W(:)-V2(:)-D2(:)).^2); 
        OUT.obj_D(iter) = obj_D;
        obj_AL  = obj_V1+obj_D;
        OUT.obj_AL(iter) = obj_AL;
        [MPSNR1,~] = MPSNR(X,M_true);
        OUT.MPSNR1(iter) = MPSNR1;
        if iter == 1 || mod(iter, 2) == 0
            fprintf('iter = %3.d, beta = %4.3f,  psnr = %3.2f,relative error = %.3f Augment Largrange obj = %.1f, Ori obj = %.1f \n',iter,beta,MPSNR1,chg,obj_AL,obj_Ori);
        end
    end
    
    if chg < tol
        break;
    end 
    %Lambda = Lambda + mu*dLam;
    beta = min(rho*beta,max_beta);    
end
obj = obj_Ori;
err = norm(M_true(:)-X(:));

 