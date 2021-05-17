function [orig, R] = computeTransPartA(proj, l1, l2, r3, toolRadius)

% The normal computed below must have the same direction
if ( (l1(1:2)'*l2(1:2)) < 0 )
    l1 = -l1;
end

% Plane through the line l1 and the optical center:
% transpose(projection matrix)^T*line
Pi1 = proj'*l1;
% Plane through the line l2 and the optical center
Pi2 = proj'*l2;

% Plane vector normalization so that the first 3 elements are of unit norm
Pi1 = normPl(Pi1);
% Plane vector normalization so that the first 3 elements are of unit norm
Pi2 = normPl(Pi2);

% Median plane between Pi1 and Pi2
P1 = (Pi1 + Pi2)/2;
P1 = normPl(P1);

% Get the plane normal (first 3 elements)
r1 = P1(1:3);

% In order to compute t, the following convention is considered:
% r1 is the plane normal
% r3 is the normal directed towards the cylinder main axis
% r2 is orthogonal to r1 and r2
r2 = unit(cross(r3,r1));
r1 = unit(cross(r2,r3));

% Flip the vector in case r2 is not directed towards the optical center
if (r2(3) > 0)
    r2 = -r2;
    r1 = -r1;
end

% Sinus of the angle formed by the two planes
sinTheta = norm(cross(P1(1:3),Pi1(1:3)));

% Distance from the camera center to the cylinder axis
d = toolRadius/sinTheta;
%d = d*1.2

% Plane through the cylinder axis whose normal is directed towards the
% optical center
P2 = [ r2 ; d ];

% Locus of points belonging to both P1 and P2
[~,~,v] = svd([P1' ; P2' ]);
A = normalize(v(:,end-1));
B = normalize(v(:,end));

% Pick up one point in the family of solutions
% The point must be in front of the camera, the third's element sign must
% be positive

if (A(3) < 0)
    orig = -A(1:3);
else
    orig = A(1:3);
end

% Update rotation
R = [ r1 r2 r3 ];
