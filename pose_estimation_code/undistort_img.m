% DATA_PATH_IMAGE = [ '/home/bao/Downloads/trocar_estimation_adrien/liver_tools_trocars_compressed/' ];
% 
% indicesToProcess = 0:5408;
% 
% imagePrefix = '%04d';
% 
% fx = 1.2046804628126897e+03;
% 
% fy = 1.2045124784966654e+03;
% 
% cx = 1.0250124702643961e+03;
% 
% cy = 6.2008270677820633e+02;
% 
% k1 = -1.1245733753016555e-01;
% 
% k2 = -7.2247216932679438e-03;
% 
% k3 = -9.1046733950792269e-03;
% 
% K = [ fx 0 cx ; 0 fy cy ; 0 0 1 ];
% 
% 
% 
% % Build camera matrix:
% 
% focal = [fx, fy];
% 
% prip = [cx, cy];
% 
% for iFrameToProcess = indicesToProcess
%     
%     I1 = imread([DATA_PATH_IMAGE 'frame' sprintf([ imagePrefix '.png' ],iFrameToProcess)]);
% 
%     imsiz = [size(I1,2),size(I1,1)];
% 
%     radial = [k1,k2,k3];
% 
%     camIntrinsics = cameraIntrinsics(focal, prip,imsiz,'RadialDistortion', radial);
% 
%     cameraParams = cameraParameters('IntrinsicMatrix',camIntrinsics.IntrinsicMatrix,'RadialDistortion',camIntrinsics.RadialDistortion);
% 
%     % Undistort images:
% 
%     IU = undistortImage(I1,cameraParams);
%     
%     imwrite(IU,[DATA_PATH_IMAGE 'undistorted/image' sprintf([ imagePrefix '.png' ],iFrameToProcess)]);
% 
% end

% ptCloud = pcread('../liver.ply');
% Mc = diag([1,-1,-1]);
% % ptCloud1 = Mc * ptCloud;
% view(3)
% pcshow(ptCloud);
% 
% hold on
% 
% a1 = load("Radius1.mat");
% b1 = struct2cell(a1);
% b1 = b1{1};
% s1 = size(b1);
% indices1 = 1:s1(2);
% pts1 = zeros(3,s1(2));
% u1 = zeros(3,s1(2));
% 
% for indice = indices1
%     pts1(:,indice) = b1{indice}(1:3,4);
%     u1(:,indice) = b1{indice}(1:3,3);
% end
% 
% vect_end1 = pts1 - 50*u1;
% vect_start1 = pts1 + 50*u1;
% 
% vect_end_x1 = vect_end1(1,:);
% vect_start_x1 = vect_start1(1,:);
% X1 = [vect_end_x1;vect_start_x1];
% 
% vect_end_y1 = vect_end1(2,:);
% vect_start_y1 = vect_start1(1,:);
% Y1 = [vect_end_y1;vect_start_y1];
% 
% vect_end_z1 = vect_end1(3,:);
% vect_start_z1 = vect_start1(1,:);
% Z1 = [vect_end_z1;vect_start_z1];
% 
% a2 = load("Radius2.mat");
% b2 = struct2cell(a2);
% b2 = b2{1};
% s2 = size(b2);
% indices2 = 1:s2(2);
% pts2 = zeros(3,s2(2));
% u2 = zeros(3,s2(2));
% 
% for indice = indices2
%     pts2(:,indice) = b2{indice}(1:3,4);
%     u2(:,indice) = b2{indice}(1:3,3);
% end
% 
% vect_end2 = pts2 - 50*u2;
% vect_start2 = pts2 + 50*u2;
% 
% vect_end_x2 = vect_end2(1,:);
% vect_start_x2 = vect_start2(1,:);
% X2 = [vect_end_x2;vect_start_x2];
% 
% vect_end_y2 = vect_end2(2,:);
% vect_start_y2 = vect_start2(1,:);
% Y2 = [vect_end_y2;vect_start_y2];
% 
% vect_end_z2 = vect_end2(3,:);
% vect_start_z2 = vect_start2(1,:);
% Z2 = [vect_end_z2;vect_start_z2];
% 
% a3 = load("Radius3.mat");
% b3 = struct2cell(a3);
% b3 = b3{1};
% s3 = size(b3);
% indices3 = 1:s3(2);
% pts3 = zeros(3,s3(2));
% u3 = zeros(3,s3(2));
% 
% for indice = indices3
%     pts3(:,indice) = b3{indice}(1:3,4);
%     u3(:,indice) = b3{indice}(1:3,3);
% end
% 
% vect_end3 = pts3 - 50*u3;
% vect_start3 = pts3 + 50*u3;
% 
% vect_end_x3 = vect_end3(1,:);
% vect_start_x3 = vect_start3(1,:);
% X3 = [vect_end_x3;vect_start_x3];
% 
% vect_end_y3 = vect_end3(2,:);
% vect_start_y3 = vect_start3(1,:);
% Y3 = [vect_end_y3;vect_start_y3];
% 
% vect_end_z3 = vect_end3(3,:);
% vect_start_z3 = vect_start3(1,:);
% Z3 = [vect_end_z3;vect_start_z3];
% 
% a4 = load("Radius4.mat");
% b4 = struct2cell(a4);
% b4 = b4{1};
% s4 = size(b4);
% indices4 = 1:s4(2);
% pts4 = zeros(3,s4(2));
% u4 = zeros(3,s4(2));
% 
% for indice = indices4
%     pts4(:,indice) = b4{indice}(1:3,4);
%     u4(:,indice) = b4{indice}(1:3,3);
% end

% vect_end4 = pts4 - 50*u4;
% vect_start4 = pts4 + 50*u4;
% 
% vect_end_x4 = vect_end4(1,:);
% vect_start_x4 = vect_start4(1,:);
% X4 = [vect_end_x4;vect_start_x4];
% 
% vect_end_y4 = vect_end4(2,:);
% vect_start_y4 = vect_start4(1,:);
% Y4 = [vect_end_y4;vect_start_y4];
% 
% vect_end_z4 = vect_end4(3,:);
% vect_start_z4 = vect_start4(1,:);
% Z4 = [vect_end_z4;vect_start_z4];
% 
% 
% a1 = line(X1,Y1,Z1,'Color','red','LineStyle','--');
% 
% a2 = line(X2,Y2,Z2,'Color','blue','LineStyle','-');
% 
% a3 = line(X3,Y3,Z3,'Color','green','LineStyle',':');
% 
% a4 = line(X4,Y4,Z4,'Color','yellow','LineStyle','-.');
% 
% M1 = "Curve 1";
% M2 = "Curve 2";
% M3 = "Curve 3";
% M4 = "Curve 4";
% 
% % view(3)
% % plot3(0, 0, 0, 'ko', 'MarkerSize', 10);
% 
% surf([93.66 7.22 -79.37 1.09], [12.91 -47.64 -12.28 -6.11 ], [-52.86 -0.035 7.02 -14.89])  
% hold off

% 
% a = load("Radius4.mat");
% b = struct2cell(a);
% b = b{1};
% s = size(b);
% indices = 1:s(2);
% % pts = zeros(3,s(2));
% % u = zeros(3,s(2));
% 
% for indice = indices
%     
%     c{indice} = b{indice}(1:3,3:4);
%     
% end
% 
% save('T4.mat','c');


% plot all tooltips for clustered to trocar 1

clear

clc

close all

% load data

print_GT = 1;

if ~print_GT

    load('gignac/prediction_result.mat');

else
    T = load('../resT1.mat','trocar1');
    T = struct2cell(T);
    T = T{1};
    c = load('../resT1.mat','inliers1');
    c = struct2cell(c);
    c = c{1};
    trocar1 = zeros(3,2,length(c));

    for i = 1:length(c)

       trocar1(:,:,i) = c(:,:,i);

    end

end

% from trocarpose_prediction.txt
% T1 = [ 14.44134314 -41.03426919 -0.32727529 ]';

% new result
% T1 = [ -52.87816382,  -4.00512916,   3.09245385 ]';
% T1 = [28.44290069, -57.32344926,   2.63260373]';
% T1 = [0.84039662,  7.20048446, -5.33116659]';
% T1 = [4.37398595,  11.24724645, -29.82472926]';
% length of axis to plot

ax_len = 10^3;

figure(1);

clf;

hold on

for i = 1:size(trocar1,3)

    if 1

        % plot the axis

        plot3(trocar1(1,2,i)*[1 1]+trocar1(1,1,i)*[0 ax_len],trocar1(2,2,i)*[1 1]+trocar1(2,1,i)*[0 ax_len],trocar1(3,2,i)*[1 1]+trocar1(3,1,i)*[0 ax_len],'b-');

        % plot the tip

        plot3(trocar1(1,2,i),trocar1(2,2,i),trocar1(3,2,i),'bo','markerfacecolor','b');

    else

        % plot the axis

        plot3(trocar1(1,1,i)*[1 1]+trocar1(1,2,i)*[0 ax_len],trocar1(2,1,i)*[1 1]+trocar1(2,2,i)*[0 ax_len],trocar1(3,1,i)*[1 1]+trocar1(3,2,i)*[0 ax_len],'b-');

        % plot the tip

        plot3(trocar1(1,1,i),trocar1(2,1,i),trocar1(3,1,i),'bo','markerfacecolor','b');

    end

end

axis equal

plot3(T(1),T(2),T(3),'ro','markerfacecolor','r');
plot3(0, 0, 0, 'ro', 'markerfacecolor', 'g');

view(3)
