function projectToolMKH(K, pose, model, color_)

if nargin < 5
    color_ = 'r';
end

RCam = pose(1:3,1:3);
tCam = pose(:,4);
proj = K*[RCam tCam];

projMKH = [RCam tCam];

% Display the coord. sys.
projectPose(proj, pose, model.radius); hold on

camCenter = -RCam'*tCam;

t = [0;0;0];
R = RfromNormal([0;0;1], t, camCenter);

alpha_ = linspace(0,1,20);
center = t;

r1 = R(:,1);
r2 = R(:,2);
% Cylinder center to camera center expressed in the
% cyl. coord. sys.
tCyl = (R'*camCenter + R'*t);
a = norm(tCyl(1:2)); % distance from the camera center to the cylinder axis (l2 distance in the x-y plane)
c = sqrt(a*a-model.radius*model.radius); % distance from the camera center to the incidence point
cosAlpha = model.radius/a; % angle between these two rays
sinAlpha = sqrt(1-cosAlpha*cosAlpha);
%beta = pi-pi/2-alpha; % the angle in which we are interested in
r3 = unit(R(:,3)); % get the cylinder direction
v1 = unit(camCenter-t);
v2 = unit(cross(r3,v1)); % normal to the plane through the camera center and the cylinder axis
v1 = unit(cross(v2,r3)); % v1 orthogonal to v2 and r3 to complete the unit matrix

v(:,1) = cosAlpha*v1 + sinAlpha*v2; % vector orthogonal to the plane tangent to the cylinder passing though the camera center
v(:,2) = cosAlpha*v1 - sinAlpha*v2;

% Create two points along the line origin + model.radius*v and
% origin + r3 + model.radius*v
%A(:,1) = normalize(proj*augment(center + model.radius*v(:,1)));
centerProj = normalize(proj*augment(center));
A(:,1) = normalize(proj*augment(center + model.radiusA*v(:,1)));
B(:,1) = normalize(proj*augment(center + model.lengthCylinderA*r3 + model.radiusA*v(:,1)));

A(:,3) = normalize(proj*augment(center + model.lengthCylinderA*r3 + model.radiusB*v(:,1)));
B(:,3) = normalize(proj*augment(center + model.lengthCylinderA*r3 + model.lengthCylinderB*r3 + model.radiusB*v(:,1)));

A(:,2) = normalize(proj*augment(center + model.radiusA*v(:,2)));
B(:,2) = normalize(proj*augment(center + model.radiusA*v(:,2) + model.lengthCylinderA*r3 ));

A(:,4) = normalize(proj*augment(center + model.lengthCylinderA*r3 + model.radiusB*v(:,2)));
B(:,4) = normalize(proj*augment(center + model.lengthCylinderA*r3 + model.radiusB*v(:,2) + model.lengthCylinderB*r3));

C = normalize(proj*augment(center));
D = normalize(proj*augment(center + model.lengthCylinderTotal*r3));

pV1 = normalize(proj*augment(center + model.radiusA*v1));

for iLine=1:4
    for ialpha_=1:length(alpha_)
        pt1 = A(:,iLine)*alpha_(ialpha_) + (1-alpha_(ialpha_))*B(:,iLine);
        pt2 = A(:,iLine)*alpha_(ialpha_) + (1-alpha_(ialpha_))*B(:,iLine);
        plot(pt1(1),pt1(2),'rx');
        plot(pt2(1),pt2(2),'rx');
        
        pt = C*alpha_(ialpha_) + (1-alpha_(ialpha_))*D;
        plot(pt(1),pt(2),'gx');
        plot(pt(1),pt(2),'gx');

    end

end

plot(pV1(1),pV1(2),'mo');

% Plot coordinate system attached to the tool
e(:,1) = model.radius*v2;
e(:,2) = model.radius*v1;
e(:,3) = -model.radius*r3;

imagedE = normalize(proj*augment(e));

color = 'rgb'

for i=1:3
   l = line( [centerProj(1), imagedE(1,i)], [centerProj(2), imagedE(2,i)]); 
   set(l,'Color',color(i),'LineWidth',4, 'LineSmoothing','on');
end

imagedTT = normalize(proj*augment(center - model.tipLength*R(:,3)))
h = plot(imagedTT(1),imagedTT(2),'r*');
set(h,'MarkerSize',15);


end
