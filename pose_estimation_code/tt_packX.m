function X = tt_packX(flags)

iParam = 0;
if flags.refineR
    X = [ 0 ; 0 ]
    iParam = iParam + 2;
end
if flags.refineT
    X(iParam+(1:3)) = [ 0 ; 0 ; 0] ;
end

X = X(:);