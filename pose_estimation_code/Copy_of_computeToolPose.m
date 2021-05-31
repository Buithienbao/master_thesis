function [] = computeToolPose(tool)

useCroppedImages = true;

%% Set the directory path of the files to estimate the pose.
% @TBM
DATA_PATH_IMAGE = [ '/home/bao/Downloads/trocar_estimation_adrien/mercuri' ];

doNonLinRefinement = 0;

% % Image indices in folder subImageA/C
% if strcmp(tool,'toolA')
%     indicesToProcess = [ 4 5 6 9 10 16 17 20 25 26 27 28 29 34 35 36 42 ];
% end
% 
% if strcmp(tool,'toolC')
%     indicesToProcess = [ 1 3 4 6 16 19 24 26 28 33 36 40 41 42 48 ];
% end
indicesToProcess = [ 43 44 49 52 61 ];
% indicesToProcess = [ 66 88 90 101 121 ];

errorRadius = [ 1 ]; % modify this line if you want to simulate noise on the the radius measure, e.g. [1.0,1.05,1.1]

% Set the maximum number of iteration for the iterative pose refinement
% (last step)
params.nIter = 30;

% In case you rescale the input image (the use of this variable must be
% checked)
params.scaleDownsize = 1;

% String used for storage when noise is applied on the ART-Net output
noiseMagnitudes = {'00','05','10','15','20','25'};

% for noiseMag=noiseMagnitudes
%     noiseMag = noiseMag{1};
%     for iError=1 % Change this line when using noise on the shaft radius measurement, e.g. 1:length(errorRadius)
        
        % You find thre measures here below, namely two radius radiusA and
        % radiusB, and tipLength.
        % * radiusA corresponds to the radius of the part of the shaft, referred to A, next
        % to the tool head (whose radius is most often larger than (or equal to) the radius of
        % the other part of the shaft. The part of the shaft whose radius
        % is radiusA is of length lengthCylinderA
        % * radiusB corresponds to the radius of the shaft after A.
        % When there is no chessboard to compute the 3D pose ground truth,
        % lengthCylinderB can be much greater than 10 mm (e.g. in [50mm-100mm])
        % and the last points used for the pose refinement end up outside
        % of the image.
        % * tipLength is the length of the tool head
        
        % There are two tools for the evaluation of animal data
% if strcmp(tool,'toolA')
%     % @TBM
%     model.radiusA = 2.4*errorRadius(iError);
%     model.radiusB = 2.25*errorRadius(iError);
%     model.lengthCylinderA = 10;
%     model.lengthCylinderB = 10;
%     model.tipLength = 13.6;
% elseif strcmp(tool,'toolC')
%     % @TBM
%     model.radiusA = 2.35*errorRadius(iError);
%     model.radiusB = 2.25*errorRadius(iError);
%     model.lengthCylinderA = 10;
%     model.lengthCylinderB = 10;
%     model.tipLength = 23.9;
% end
model.radiusA = 2.5;
model.radiusB = 2.4;
model.lengthCylinderA = 10;
model.lengthCylinderB = 10;
model.tipLength = 35;
% The mean of radiusA and radiusB is used as the best approximation 
% of the tool radius in the pose initialisation
model.radius = (model.radiusA+model.radiusB)/2;

model.lengthCylinderTotal = model.lengthCylinderA + model.lengthCylinderB;

params.nSample = 20; % nb sample points along the cylindrical parts
                     % used in the pose refinement

% For result storage                   
imagePrefix = '%02d';
% imagePrefix = '%03d';

% % Load the sub images coordinates
% % @TBM
% load(['../2nd_session/undist_images/' tool '/subImages/data.mat']);
% 
% % @TBM (if the result already exists, it will be updated)
% if exist(['../2nd_session/undist_images/' tool '/subImages/results' 'Radius' int2str(iError) '_noise_' noiseMag '.mat'],'file')
%     load(['../2nd_session/undist_images/' tool '/subImages/results' 'Radius' int2str(iError) '_noise_' noiseMag '.mat']);
% end

frIds2 = [ 0 1 2 3 4 ];

refinePoseToSave = {};
% Loop over all the image in a folder
for ite = 1:10000
    
    id1 = 1;

    for iFrameToProcess = indicesToProcess
        disp("____________________")
    %     Iref = uint8(zeros(1080,1920,1));
    %     subImage = subImages(iFrameToProcess);

        % If useCroppedImages, then the use of the cropped images will
        % be used. Cropped images are used for the quantitative
        % evaluation (see figure 16 of the MIA paper)
        % @TBM
    %     if ~useCroppedImages
    %         I = imread([DATA_PATH_IMAGE '/' sprintf([ imagePrefix '.png' ],iFrameToProcess)]);
    %         I_bothline = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_edgeLine' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    %         I_midline = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_midline' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    %         I_tippoint = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_tipPoint_Approximated' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    %         I_Line1 = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_edgeLine_Line_1' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    %         I_Line2 = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_edgeLine_Line_2' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    %     else
    %         I = imread(subImages(iFrameToProcess).rawImagePath);
    %         I_bothline_sub = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_edgeLine' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    %         I_midline_sub = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_midline' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    %         I_tippoint_sub = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_tipPoint_Approximated' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    %         I_Line1_sub = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_edgeLine_Line_1' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    %         I_Line2_sub = imread([DATA_PATH_IMAGE '/results' '_noise/' sprintf([ imagePrefix '_edgeLine_Line_2' '_noise_' noiseMag '.png'], iFrameToProcess) ]);
    % 
    %         I_bothline = Iref;
    %         I_bothline(subImage.position(2):(subImage.position(2)+subImage.position(4)),subImage.position(1):(subImage.position(1)+subImage.position(3)),:) = I_bothline_sub;
    %         I_midline = Iref;
    %         I_midline(subImage.position(2):(subImage.position(2)+subImage.position(4)),subImage.position(1):(subImage.position(1)+subImage.position(3)),:) = I_midline_sub;
    %         I_tippoint = Iref;
    %         I_tippoint(subImage.position(2):(subImage.position(2)+subImage.position(4)),subImage.position(1):(subImage.position(1)+subImage.position(3)),:) = I_tippoint_sub;
    %         I_Line1 = Iref;
    %         I_Line1(subImage.position(2):(subImage.position(2)+subImage.position(4)),subImage.position(1):(subImage.position(1)+subImage.position(3)),:) = I_Line1_sub;
    %         I_Line2 = Iref;
    %         I_Line2(subImage.position(2):(subImage.position(2)+subImage.position(4)),subImage.position(1):(subImage.position(1)+subImage.position(3)),:) = I_Line2_sub;
    %     end

        I = imread([DATA_PATH_IMAGE '/undistorted/' 'image' sprintf([ imagePrefix '.png' ],iFrameToProcess)]);
        I_bothline = imread([DATA_PATH_IMAGE '/results/' 'image' sprintf([ imagePrefix '_edgeLine.png'], iFrameToProcess) ]);
        I_midline = imread([DATA_PATH_IMAGE '/results/' 'image' sprintf([ imagePrefix '_midline.png'], iFrameToProcess) ]);
        I_tippoint = imread([DATA_PATH_IMAGE '/results/' 'image' sprintf([ imagePrefix '_tipPoint_Approximated.png'], iFrameToProcess) ]);
        I_Line1 = imread([DATA_PATH_IMAGE '/results/' 'image' sprintf([ imagePrefix '_edgeLine_Line_1.png'], iFrameToProcess) ]);
        I_Line2 = imread([DATA_PATH_IMAGE '/results/' 'image' sprintf([ imagePrefix '_edgeLine_Line_2.png'], iFrameToProcess) ]);
        % Calibration matrix
        K = [ 952.06,       0,  884.47
            0, 952.06,  553.68
            0,       0,    1.00];

        pose = [ eye(3) zeros(3,1) ];
        proj = K*pose;

        %% Specify the actions to be run and some necessary parameters
        % Set some of this variable according to the intermediate
        % results you would like to see.
        flags.displayLineDetection = 0;
        flags.displayPose = 0;
        flags.displayPoseInit = 0;
        flags.displayImages = 0;
        flags.doSave = 1;

        tic

        %% Display all original Image with the geometric features
        if flags.displayImages
            figure(1)
            subplot 321, imshow(I), title('Original Image')
            subplot 322, imshow(I_bothline), title('BothLine')
            subplot 323, imshow(I_midline), title('MidLine')
            subplot 324, imshow(I_tippoint), title('TipPoint')
            %pause
        end

        %% Finding the co-ordinate of the tip point.
        [CK,IP] = max(I_tippoint(:)); % tip point as the max of the ART-Net map
        [I1,I2,I3,I4] = ind2sub(size(I_tippoint),IP);
        TipPoint=[I2;I1;1];


        %% Mid-Line detection
        I_midline_Threshold = I_midline;
        % Threshold the ART-Net map before applying the hough
        % transform
        I_midline_Threshold(I_midline_Threshold<30)=0;

        [H,theta,rho] = hough(I_midline_Threshold);
        P = houghpeaks(H,1,'threshold',ceil(0.3*max(H(:))));
        lines_mid = houghlines(I_midline_Threshold,theta,rho,P(1,:),'FillGap',5,'MinLength',7);

        % If no line is detected, jump to the next frame
        if isempty(lines_mid)
            continue;
        end

        if flags.displayLineDetection
            figure(2)
            imshowpair(I_midline,I), title('MidLineApprox+MidLinePred+Original')
            hold on;
            displayLines(lines_mid);
            'Line mid'
            pause
        end

        l_mid = cross(augment(lines_mid(1).point1'), augment(lines_mid(1).point2'));
        % Normalize so that the scalar product give the point-to-line euclidean distance
        l_mid = l_mid/norm(l_mid(1:2))


        %% Line_1 detection
        [H,theta,rho] = hough(I_Line1);
        P = houghpeaks(H,1,'threshold',ceil(0.3*max(H(:))));
        lines_1 = houghlines(I_Line1,theta,rho,P(1,:),'FillGap',5,'MinLength',7);

        if flags.displayLineDetection
            figure(3)
            imshowpair(I_Line1,I), title('Line1+Original')
            hold on;
            displayLines(lines_1);
            'Line 1'
            pause
        end

        % The first edge line is computed using the cross product of
        % two points lying on it using their augmented cartesian
        % coordinates
        l_line_1 = cross(augment(lines_1(1).point1'), augment(lines_1(1).point2'));
        % Normalize so that the scalar product give the point-to-line euclidean distance
        l_line_1 = l_line_1/norm(l_line_1(1:2))

        %% Line_2 detection
        [H,theta,rho] = hough(I_Line2);
        P = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
        lines_2 = houghlines(I_Line2,theta,rho,P(1,:),'FillGap',5,'MinLength',7);


        if flags.displayLineDetection
            figure(4)
            imshowpair(I_Line2,I), title('Line2+Original')
            hold on;
            displayLines(lines_2);
            'Line 2'
            pause
        end

        % The first edge line is computed using the cross product of
        % two points lying on it using their augmented cartesian
        % coordinates
        l_line_2 = cross(augment(lines_2(1).point1'), augment(lines_2(1).point2'));
        % Normalize so that the scalar product give the point-to-line euclidean distance
        l_line_2 = l_line_2/norm(l_line_2(1:2))


        %%
        %%%%%%%%%%%%%%%%%%%%%%%%
        %   Pose computation   %
        %%%%%%%%%%%%%%%%%%%%%%%%

        % A. Compute and calibrate the vanishing point intersection of the two
        % tangent lines
        % B. Get a rotation matrix out of SO3 whose r3 is equal to the estimated
        % point at infinity
        % C. Compute the translation term up to a translation along the cylinder
        % axis.
        % D. Solve the latter ambiguity with radial point lying on the 3D circles
        % deliminated the rings
        % E. Refine the entire pose based on the segmented edge points obtained
        % above

        % A. Compute r3`
        % The vanishing point vanishP as the cross product of the two
        % edge line
        vanishP = cross(l_line_1,l_line_2);

        % A second option consists of using the three lines by
        % computing the point that minimizes the distance to the three
        % lines. It can be achieved using the below linear least squares
        % solution (step 4, table 1 of the MIA paper)
        % [~,~,V] = svd([l_line_1';l_mid';l_line_2']);
        % vanishP = V(:,end);

        % Ideal position
        r3 = unit(K\vanishP);

        %%

        % Keep this "if" if you work on the affine ambiguity problem when the tool
        % is e.g. far from camera or its orientation nearly parallel to
        % the pixel plane.

        %for iAffine=1:2


        % B. Compute R
        % Rank-1 matrix from r3
        rankOneRot = r3*r3';
        % Pick a rotation matrix that "contains" r3
        [U,S,V] = svd(rankOneRot);
        % Place r3 as the last column
        U = U(:,[2 3 1]);

        % Set rotational part of the pose to U
        R = U;

        % Check the direction of r3
        if R(3,3)>0
            R(:,1) = -R(:,1);
            R(:,3) = -R(:,3);
        end

        %%
        % C. Compute t up to a scalar lambda along the cylinder axis
        % -> tangent constraints while the only unknown are t1, t2, t3 and the
        % rotational part is known

        % Compute a transformation used to condition the image data.
        % Check the function content for details.
        [T, invT] = vgg_conditioner_from_image(size(I,2), size(I,1), K(1,1));

        % Convention used for the camera pose
        pose = [ eye(3) zeros(3,1) ];

        % Conditioned projection matrix
        projCond = T*K*pose;

        % Condition the lines. The inverse of T must be applied in the
        % dual projective space.
        l1 = inv(T)'*l_line_2;
        l2 = inv(T)'*l_line_1;

        % Compute the translation term of the pose based on the edge
        % lines, the camera pose, the estimated r3 and the shaft radius
        [toolOrig,R] = computeTransPartA(projCond, l1, l2, R(:,3), model.radius);

        if flags.displayPoseInit
            figure(5);
            imshow(I,'InitialMagnification','fit');
            hold on;
            projectPose(proj, [ R toolOrig ], model.radius);
            pause
        end

        %%
        % Compute t3
        clear t1 t2 t3
        % Set t1 and t2 at their computed values
        t1 = toolOrig(1);
        t2 = toolOrig(2);

        if flags.displayPoseInit
            figure(6);
            imshow(I,'InitialMagnification','fit');
            hold on;
            plot(TipPoint(1),TipPoint(2),'go');
            pause
        end

        %% Compute lambda such that the tool position 
        % t = toolOrig + lambda*r3

        % This part does not match with the item (5) of the table 1 of
        % the MIA paper. I encourage the use of item (5) here.

        signeZ = sign(R(3,2));
        % Univariate linear equation to retrieve t3
        % 3 constraints, namely cross product between the measured point and its reprojection
        linSys = [ (K(1,1)*R(1,3) - TipPoint(1)*R(3,3) + K(1,3)*R(3,3)) ...
            (K(1,1)*(toolOrig(1) - R(1,2)*signeZ*model.radius) - TipPoint(1)*(toolOrig(3) - R(3,2)*signeZ*model.radius) + K(1,3)*(toolOrig(3) - R(3,2)*signeZ*model.radius))
            (K(2,2)*R(2,3) - TipPoint(2)*R(3,3) + K(2,3)*R(3,3)) ...
            (K(2,2)*(toolOrig(2) - R(2,2)*signeZ*model.radius) - TipPoint(2)*(toolOrig(3) - R(3,2)*signeZ*model.radius) + K(2,3)*(toolOrig(3) - R(3,2)*signeZ*model.radius))];

        [~,~,VV] = svd(linSys);
        sol = VV(:,end);
        sol = sol(1)/sol(2);

        toolOrig = toolOrig+sol*R(:,3);

        if flags.displayPoseInit
            figure(7)
            imshow(I,'InitialMagnification','fit');
            hold on;
            projectToolMKH(K,[R toolOrig], model);
            disp(['iFrameToProcess: ' int2str(iFrameToProcess)]);
            pause
        end


        % I kept this block which comes along with the if located line
        % 271. If the if and the below block are uncommented, you will
        % have two solutions. Using tracking frame to frame or any
        % other prior information may be used to select the good
        % solution.

        %         % Move to the 2nd affine case
        %         nTmp = R(:,3);
        %         rB = unit(-toolOrig);
        %         rC = unit(cross(nTmp, rB));
        %         rA = unit(cross(rB,rC));
        %         rC = cross(rA,rB);
        %         e1 = nTmp'*rA;
        %         e2 = nTmp'*rB;
        %         r3 = e1*rA-e2*rB;
        %
        %     end

        pose = [ R toolOrig ];

        disp(pose);

        M = load([DATA_PATH_IMAGE sprintf('/exported_models/ABSOR_pose_view%d.txt',frIds2(id1))]);

        M = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1] * M;

        pose1 = pose * M;

        disp(pose1);

        refinePoseToSave{5*(ite-1)+id1} = pose1;

        id1 = id1 + 1;

        %% Perform the pose refinement according to the item 6 of table 1 of the MIA paper
        if doNonLinRefinement

            % E. Non linear refinement of the pose
            pose = [ R toolOrig ];

            disp('Pose refinement');
            flags.refineT = 1;
            flags.refineR = 1;
            flags.costTangent = 1;
            flags.costRadial = 0;
            flags.geomCost = 0;
            flags.photomCost = 1;
            flags.displayA_ = 0;
            flags.displayB_ = 0;
            flags.debug = 0;

            IMG.imgBound = I_bothline;
            IMG.imgTip = I_tippoint;
            IMG.imgMid = I_midline;

            IMG.offset = [ 0 ; 0 ];

            disp('Starts refinement');

            % Call the refinement function. The returned solution is a
            % non linear least squares solution computed using an
            % iterative optimization (levenberg-marquardt algorithm)
            poseRes = refinePoseMKH(IMG, K, pose, model, flags, params, params.nIter);

            if flags.displayPose
                figure(8);
                imshow(I); hold on;
                projectToolMKH(K,poseRes, model);
                hold off;
                pause
            end

            poseResNew = poseRes;
            disp('poseResNew')
            disp(poseResNew)

            refinePoseToSave{iFrameToProcess} = poseRes;
        end


%         % @TBM
%         if flags.doSave
%             save([DATA_PATH_IMAGE '/results/Radius.mat'],'refinePoseToSave');
%         end
    end
end

% @TBM
if flags.doSave
    save([DATA_PATH_IMAGE '/results/Radius.mat'],'refinePoseToSave');
end
%     end
% end

