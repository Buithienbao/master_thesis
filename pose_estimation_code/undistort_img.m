DATA_PATH_IMAGE = [ '/home/bao/Downloads/trocar_estimation_adrien/mercuri/' ];

indicesToProcess = [ 43 44 49 52 61 ];

imagePrefix = '%02d';

fx = 9.5206967601396957e+02;

fy = 9.5206967601396957e+02;

cx = 8.8447392341747241e+02;

cy = 5.5368748726315528e+02;

k1 = -2.3184898303739043e-01;

k2 = 3.4051349187505525e-01;

k3 = -2.3860036358921605e-01;

K = [ fx 0 cx ; 0 fy cy ; 0 0 1 ]



% Build camera matrix:

focal = [fx, fy];

prip = [cx, cy];

imsiz = [size(I1,2),size(I1,1)];

radial = [k1,k2,k3];

camIntrinsics = cameraIntrinsics(focal, prip,imsiz,'RadialDistortion', radial);

cameraParams = cameraParameters('IntrinsicMatrix',camIntrinsics.IntrinsicMatrix,'RadialDistortion',camIntrinsics.RadialDistortion);

for iFrameToProcess = indicesToProcess

    I1 = imread([DATA_PATH_IMAGE 'image' sprintf([ imagePrefix '.png' ],iFrameToProcess)]);

    % Undistort images:

    IU = undistortImage(I1,cameraParams);
    
    imwrite(IU,[DATA_PATH_IMAGE 'undistorted/image' sprintf([ imagePrefix '.png' ],iFrameToProcess)]);

end
