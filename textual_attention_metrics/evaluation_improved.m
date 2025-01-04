function [evals] = evalution_improved(template, imgA, imgB, imgF)

evals(1) = analysis_Qabf_improved(template,imgB,imgF); 
evals(2) = analysis_ssim_improved(template,imgB,imgF); 
evals(3)  = vifvec(im2double(template),im2double(imgF));

end
