function R = RfromNormal(n, t, camCenter)

n = unit(n);

if (nargin == 1)
    % Compute a rotation matrix whose 3rd direction is n
    rankOneRot = n*n';
    
    [U,S,V] = svd(rankOneRot);
    
    U = U(:,[2 3 1]);
    % Always directed toward z (??)
    if (U(:,3)'*[0;0;1] < 0)
        U(:,3) = -U(:,3);
        % Be sure that we stay in a direct coordinate system
        if (det(U) < 0)
            U(:,1) = -U(:,1);
        end
    end
    R = U; 
else
    % Compute a rotation matrix whose 3rd direction is n
    r3 = n;
    r2 = unit(camCenter-t(:));
    r1 = unit(cross(r2,r3));
    r2 = unit(cross(r3,r1));
    R = [r1 r2 r3];
end