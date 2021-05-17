function projectPose(proj, pose, radius, iColorSet)

if nargin < 3
    radius = 1;
end

if nargin < 4
    iColorSet = 1;
end

color = 'rgbcym';

R = pose(1:3,1:3);
t = pose(:,4);

c = normalize(proj*augment(t));

e(:,1) = t + radius*R(:,1);
e(:,2) = t + radius*R(:,2);
e(:,3) = t + radius*R(:,3);

imagedE = normalize(proj*augment(e));

for i=3:-1:1
   l = line( [c(1), imagedE(1,i)], [c(2), imagedE(2,i)]); 
   set(l,'Color',color((iColorSet-1)*3+i),'LineWidth',4, 'LineSmoothing','on');
end