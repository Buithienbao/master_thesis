function res = tt_costFunctionMKH(X, flags, model, params, IMG, K, RCam, tCam, nOrig, tOrig, mask)

res = [];

proj = K*[RCam tCam]; % Camera pose

camCenter = -RCam'*tCam; % Camera center in the world frame

[R, t] = tt_unpackX(X, nOrig, tOrig,-RCam'*tCam, flags); % From parameters to camera pose

if (flags.displayA_ || flags.displayB_)
    figure(7);
    imagesc(IMG.imgBound); axis equal ; hold on;
    color_ = 'rgmc'
end

alpha_ = linspace(0,1, params.nSample);
center = t;

r1 = R(:,1);
r2 = R(:,2);

% Cylinder center to camera center expressed in the cyl. coord. sys.
tCyl = (R'*camCenter + R'*t);
a = norm(tCyl(1:2)); % distance from the camera center to the cylinder axis (l2 distance in the x-y plane)
c = sqrt(a*a-model.radius*model.radius); % distance from the camera center to the incidence point
cosAlpha = model.radius/a; % angle between these two rays
sinAlpha = sqrt(1-cosAlpha*cosAlpha);

%beta = pi-pi/2-alpha; % the angle in which we are interested in
r3 = unit(R(:,3)); % get the cylinder direction
v1 = unit(camCenter-t);
v2 = unit(cross(r3,v1));% normal to the plane through the camera center and the cylinder axis
v1 = unit(cross(v2,r3));% v1 orthogonal to v2 and r3 to complete the unit matrix

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

pV1 = normalize(proj*augment(center + model.radiusA*v1));
pV2 = normalize(proj*augment(center + model.radiusA*v2));

if flags.displayB_
    
    line([A(1,1) B(1,1)]-IMG.offset(1), [A(2,1) B(2,1)]-IMG.offset(2), 'Color', color_(1));
    plot(A(1,1)-IMG.offset(1),A(2,1)-IMG.offset(2),'r*','MarkerSize',10);
    plot(B(1,1)-IMG.offset(1),B(2,1)-IMG.offset(2),'g*','MarkerSize',10);
    plot(centerProj(1)-IMG.offset(1),centerProj(2)-IMG.offset(2),'bo','MarkerSize',10);
    line([A(1,1) centerProj(1)]-IMG.offset(1), [A(2,1) centerProj(2) ]-IMG.offset(2), 'Color', color_(3));
    line([centerProj(1) pV1(1)]-IMG.offset(1), [centerProj(2) pV1(2)]-IMG.offset(2), 'Color', color_(1));
    line([centerProj(1) pV2(1)]-IMG.offset(1), [centerProj(2) pV2(2)]-IMG.offset(2), 'Color', color_(2));
end

l(:,1) = cross(A(:,1),B(:,1));
l(:,1) = l(:,1)/norm(l(1:2,1));

% The second tanget plane/line
A(:,2) = normalize(proj*augment(center + model.radiusA*v(:,2)));
B(:,2) = normalize(proj*augment(center + model.radiusA*v(:,2) + model.lengthCylinderA*r3 ));

A(:,4) = normalize(proj*augment(center + model.lengthCylinderA*r3 + model.radiusB*v(:,2)));
B(:,4) = normalize(proj*augment(center + model.lengthCylinderA*r3 + model.radiusB*v(:,2) + model.lengthCylinderB*r3));


C = normalize(proj*augment(center));
D = normalize(proj*augment(center + model.lengthCylinderTotal*r3));

if flags.displayB_
    plot(A(1,2)-IMG.offset(1),A(2,2)-IMG.offset(2),'r*','MarkerSize',10);
    plot(B(1,2)-IMG.offset(1),B(2,2)-IMG.offset(2),'g*','MarkerSize',10);
    line([A(1,2) B(1,2)]-IMG.offset(1), [A(2,2) B(2,2)]-IMG.offset(2), 'Color', color_(2));
    line([A(1,2) centerProj(1)]-IMG.offset(1), [A(2,2) centerProj(2) ]-IMG.offset(2), 'Color', color_(3));
end

l(:,2) = cross(A(:,2),B(:,2));
l(:,2) = l(:,2)/norm(l(1:2,2));

width = size(IMG.imgBound,2);
height = size(IMG.imgBound,1);

if flags.photomCost
    for iLine=1:4
        for ialpha_=1:length(alpha_)
            pt1 = A(:,iLine)*alpha_(ialpha_) + (1-alpha_(ialpha_))*B(:,iLine);
            pt2 = A(:,iLine)*alpha_(ialpha_) + (1-alpha_(ialpha_))*B(:,iLine);
            
            % Transform back to the original image coordinate
            
            pt1(1:2) = pt1(1:2) - IMG.offset;
            pt2(1:2) = pt2(1:2) - IMG.offset;
            
            pt1(1) = max(2,min(width-1,pt1(1)));
            pt1(2) = max(2,min(height-1,pt1(2)));
            pt2(1) = max(2,min(width-1,pt2(1)));
            pt2(2) = max(2,min(height-1,pt2(2)));
            
            if flags.displayB_
                plot(pt1(1),pt1(2),'rx');
                plot(pt2(1),pt2(2),'rx');
                %pause
            end
            
            if pt1(1)<width && pt1(1)>0 && pt1(2)<height && pt1(2)>0
                
                sigPt = interp2(double(IMG.imgBound),pt1(1),pt1(2));
                sigPt = 1-sigPt/255;

                res = [ res ; sigPt ];
            end
            
            if isnan(res(end))
                res
                error('NaN in the residual, tangential part');
            end
        end
    end
    
    if (flags.displayA_ || flags.displayB_)
        figure(8);
        imagesc(IMG.imgMid); axis equal ; hold on;
    end
    
    for ialpha_=1:length(alpha_)
        pt = C*alpha_(ialpha_) + (1-alpha_(ialpha_))*D;
        
        % Transform back to the original image coordinate
        pt(1:2) = pt(1:2) - IMG.offset;
        
        pt(1) = max(2,min(width-1,pt(1)));
        pt(2) = max(2,min(height-1,pt(2)));
        
        if flags.displayB_
            plot(pt(1),pt(2),'gx');
            plot(pt(1),pt(2),'gx');
            %pause
        end
        
        if pt1(1)<width && pt1(1)>0 && pt1(2)<height && pt1(2)>0
            sigPt = interp2(double(IMG.imgMid),pt(1),pt(2));
            sigPt = 1-sigPt/255;
            res = [ res ; sigPt];
        end
        
        if isnan(res(end))
            res
            error('NaN in the residual, tangential part');
        end
    end
    
    if pV1(1)<width && pV1(1)>0 && pV1(2)<height && pV1(2)>0
        sigPt = interp2(double(IMG.imgTip),pV1(1),pV1(2));
        sigPt = 1-sigPt/255;
        res = [ res ; sigPt];
    end
end

if ( flags.displayA_ || flags.displayB_ )
    pause
end

if flags.debug
    res
end
