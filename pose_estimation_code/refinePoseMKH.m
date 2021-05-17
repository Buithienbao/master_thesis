function poseRes = refinePoseMKH(IMG, K, pose, model, flags, params, nIterMax)

if ( flags.displayA_ || flags.displayB_ )
    addpath([ MTWP_PATH '/common/Lilian/' ]);
end

options = optimset( ...
    'Jacobian','off',...
    'LargeScale','off',...
    'DerivativeCheck','off',...
    'MaxIter',nIterMax,...
    'MaxFunEvals',10e7,...
    'Display','iter',...
    'Algorithm','levenberg-marquardt');

% Initial parameters
nOrig = pose(:,3);
tOrig = pose(:,4);

RCam = RfromNormal(nOrig, tOrig, zeros(3,1));
tCam = tOrig;

% Packing all the model parameters
X0 = tt_packX(flags);

costFunction = @(X)tt_costFunctionMKH(X, flags, model, params, IMG, K, RCam, tCam, [0;0;1], [0;0;0]);

Xsol = lsqnonlin(costFunction,X0(:),[],[],options);

% Unpacking parameters
camCenter = -RCam'*tCam;
[RRes, tRes] = tt_unpackX(Xsol, [0;0;1], [0;0;0], camCenter, flags);

% Resulting pose
poseRes = [ RRes tRes ];

poseRes = [ RCam tCam ; 0 0 0 1 ]*[ poseRes ; 0 0 0 1 ];
poseRes = poseRes(1:3,:);

