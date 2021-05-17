clear all

% Two tools were processed, called toolA and toolC (cf. paper for details)
tools = {'toolA','toolC'};

% Calibration matrix
K = [ 1510.83,       0,  960.16
    0,       1510.83,  450.77
    0,       0,        1.00    ];

% Enable the display of some results for qualitative evaluation
display2D = 0;
display3D = 0;

% Noise magnitude displayed in the legend
% They correspond to 0.0, 0.5, 1.0, 1.5, 2.0 and 2.5 which is the std of
% the gaussian noise applied on the ART-Net outputs.
noiseMagnitudes = {'00','05','10','15','20','25'};

for iNoiseMag=1:length(noiseMagnitudes)
    noiseMag = noiseMagnitudes{iNoiseMag};
    
    cnt = 1;
    
    for iError=1
        for tool = tools
            tool = tool{1};
            
            if strcmp(tool,'toolA')
                model.radiusA = 2.4;
                model.radiusB = 2.25;
                model.lengthCylinderA = 10;
                model.lengthCylinderB = 10;
                model.tipLength = 13.6;
            elseif strcmp(tool,'toolC')
                model.radiusA = 2.35;
                model.radiusB = 2.25;
                model.lengthCylinderA = 10;
                model.lengthCylinderB = 10;
                model.tipLength = 23.9;
            end
            model.radius = (model.radiusA+model.radiusB)/2;
            model.lengthCylinderTotal = model.lengthCylinderA + model.lengthCylinderB;
            
            % @TBM
            load(['../2nd_session/undist_images/' tool '/subImages/data.mat']);
            load(['../2nd_session/undist_images/' tool '/subImages/results' 'Radius' int2str(iError) '_noise_' noiseMag '.mat']);
            
            if strcmp(tool,'toolA')
                indicesToProcess = [ 4 5 6 9 10 16 17 20 25 26 27 28 29 34 35 36 42 ];
                tipLength = 13.6;
            end
            
            if strcmp(tool,'toolC')
                indicesToProcess = [ 1 3 4 6 16 19 24 26 28 33 36 40 41 42 48 ];
                tipLength = 23.9;
            end
            
            for i=indicesToProcess
                
                poseEst = refinePoseToSave{i}
                
                imageNumber = str2num(subImages(i).rawImagePath(end-8:end-4));
                
                % @TBM
                gt = load(sprintf(['../2nd_session/undist_images/' tool '/subImages/GT/%05d.mat'],imageNumber))
                
                % Error computation
                origEst = poseEst(:,4);
                errorOrig(:,cnt) = abs(gt.ToolTip2-origEst)
                ttEst = poseEst(:,4) - tipLength*poseEst(:,3);
                errorTT(:,cnt) = abs(gt.ToolTip1-ttEst)
                errorAngle(cnt) = acos(-gt.UV1'*poseEst(:,3))*180/pi
                cnt = cnt+1;
                
                poseGT = [gt.UV2 gt.UV3 gt.UV1 gt.ToolTip2];
                
                % Visual results
                % Reprojection errors
                if display2D
                    %figure;
                    I = imread(subImages(i).rawImagePath);
                    %imshow(I);
                    system(sprintf('mkdir video/%s/%02d',tool,i));
                    %saveas(gca,sprintf(['video/' tool '/%02d/rawImage.png'],i));
                    pos = subImages(i).position;
                    mask = 0.6*ones(size(I));
                    subImage = I(pos(2):pos(2)+pos(4),pos(1):pos(1)+pos(3),:);
                    mask(pos(2):pos(2)+pos(4),pos(1):pos(1)+pos(3),:) = 1;
                    I = double(I).*mask;
                    I = uint8(I);
                    figure;
                    imshow(I);
                    rectangle('Position',pos, 'Linewidth',1);
                    %pause
                    saveas(gca,sprintf(['video/' tool '/%02d/maskedImage.png'],i));
                    figure;
                    imshow(subImage); hold on;
                    Kaux = K;
                    Kaux(1,3) = K(1,3)-pos(1);
                    Kaux(2,3) = K(2,3)-pos(2);
                    projectToolMKH(Kaux, poseEst, model);
                    saveas(gca,sprintf(['video/' tool '/%02d/reproj.png'],i));
                end
                
                % Visual results
                % 3D error visualisation
                if display3D
                    figure;
                    cam = plotCamera('Location',[0 0 0],'Orientation',eye(3),'Opacity',0, 'Size',5); hold on;
                    poseEst(:,1) = -poseEst(:,1);
                    poseEst(:,3) = -poseEst(:,3);
                    display3DTool(model,poseEst,'b');
                    display3DTool(model,poseGT,'g');
                    
                    axis equal;
                    view([0,-90]);
                    xlim([-40 40]);
                    ylim([-25 25]);
                    box on;
                    grid on;
                    xlabel('X (mm)');
                    ylabel('Y (mm)');
                    saveas(gca,sprintf(['video/' tool '/%02d/camView.png'],i));
                    %pause
                    view([0,0]);
                    zlim([-5 140]);
                    grid on;
                    ylabel('Y (mm)');
                    zlabel('Z (mm)');
                    saveas(gca,sprintf(['video/' tool '/%02d/topView.png'],i));
                end
            end
        end
    end
    
    % Store errors on
    % the origin location
    eO(iNoiseMag,:) = sqrt(sum(errorOrig.*errorOrig,1));
    % the tool tip location
    eT(iNoiseMag,:) = sqrt(sum(errorTT.*errorTT,1));
    % the shaft orientation
    eA(iNoiseMag,:) = errorAngle;
    
end

sizeFontLabel = 15;

figure;
h = boxplot(eO',[0,0.5,1.0,1.5,2.0,2.5]);
set(h,'LineWidth',2);
xlabel('std of the noise applied on the ART-Net maps', 'FontSize', sizeFontLabel);
ylabel('position error of the origin (mm)', 'FontSize', sizeFontLabel);
grid on;
ylim([0 15]);
pause

figure;
h = boxplot(eT',[0,0.5,1.0,1.5,2.0,2.5]);
set(h,'LineWidth',2);
xlabel('std of the noise applied on the ART-Net maps', 'FontSize', sizeFontLabel);
ylabel('position error of the tip (mm)', 'FontSize', sizeFontLabel);
grid on;
ylim([0 20]);
pause

figure;
h = boxplot(eA',[0,0.5,1.0,1.5,2.0,2.5]);
set(h,'LineWidth',2);
xlabel('std of the noise applied on the ART-Net maps', 'FontSize', sizeFontLabel);
ylabel('angular error of the tool shaft (degrees)', 'FontSize', sizeFontLabel);
grid on;
ylim([0 60]);

