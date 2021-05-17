function [R, t] = tt_unpackX(X, nOrig, tOrig, camCenter, flags)

r3 = zeros(3,1);
t = zeros(3,1);

iParam = 0;
if flags.refineR
    r3 = [ sin(X(1))*cos(X(2)) ; sin(X(1))*sin(X(2)) ; cos(X(1)) ];
    iParam = iParam + 2;
else
    r3 = nOrig;
end
if flags.refineT
    t = X(iParam+(1:3));
else
    t = tOrig;
end

R = RfromNormal(r3, t, camCenter);