DATA_PATH_IMAGE = [ '/home/bao/Downloads/trocar_estimation_adrien/finat/' ];

indicesToProcess = [ 66 88 90 101 121 ];

imagePrefix = '%03d';

fx = 1.9627645283697193e+03;

fy = 1.9569201703481888e+03;

cx = 1.0457889350948246e+03;

cy = 4.7855060990037157e+02;

k1 = -2.5094556282958186e-01;

k2 = 4.7288007862790116e-01;

k3 = -4.4664893111535614e-01;

K = [ fx 0 cx ; 0 fy cy ; 0 0 1 ];



% Build camera matrix:

focal = [fx, fy];

prip = [cx, cy];

for iFrameToProcess = indicesToProcess
    
    I1 = imread([DATA_PATH_IMAGE 'image' sprintf([ imagePrefix '.png' ],iFrameToProcess)]);

    imsiz = [size(I1,2),size(I1,1)];

    radial = [k1,k2,k3];

    camIntrinsics = cameraIntrinsics(focal, prip,imsiz,'RadialDistortion', radial);

    cameraParams = cameraParameters('IntrinsicMatrix',camIntrinsics.IntrinsicMatrix,'RadialDistortion',camIntrinsics.RadialDistortion);

    % Undistort images:

    IU = undistortImage(I1,cameraParams);
    
    imwrite(IU,[DATA_PATH_IMAGE 'undistorted/image' sprintf([ imagePrefix '.png' ],iFrameToProcess)]);

end
