function [evals] = evalution_original(imgA, imgB, imgF)

evals(1) = analysis_Qabf(imgA,imgB,imgF);
evals(2) = analysis_ssim(imgA,imgB,imgF); 
evals(3)  = (vifvec(im2double(imgA),im2double(imgF))+vifvec(im2double(imgB),im2double(imgF)))/2;

end
