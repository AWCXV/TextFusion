clear;
%p = parpool('local',6); for parallel calculation
currentFolder = pwd;
addpath(genpath(currentFolder));

tot = 2;
load('IVT_TNO_test30_textConfidence.mat');

method = 'TextFusion'
fused_path = strcat("outputs_TextFusion_TNO30/");

%parfor i =1:tot , for parallel calculation
for i =1:tot

	disp(['image:',num2str(i)]);			
	source_ir  = ['IVT_test_TNO/ir/',num2str(i),'.png'];			
	source_vis  = ['IVT_test_TNO/vis/',num2str(i),'.png'];
	text_guided_image_path = ['./reference_text_guided_images/TNO_test30/fuse',num2str(i),'.jpg'];
	image1 = imread(source_ir);
	image2 = imread(source_vis);
	text_guided_image = imread(text_guided_image_path);
	
	if size(image1, 3) == 3
		image1 = rgb2gray(image1);
	end
	if size(image2, 3) == 3
		image2 = rgb2gray(image2);
	end		
	if size(text_guided_image, 3) == 3
		text_guided_image = rgb2gray(text_guided_image);
	end		
	
	try
		fused   = strcat(fused_path,"/",num2str(i),".png");
		fused_image   = imread(fused);
	catch
		fused   = strcat(fused_path,"/",num2str(i),".jpg");
		fused_image   = imread(fused);
	end
	if size(fused_image, 3) == 3
		fused_image = rgb2gray(fused_image);
	end
	
	fused_image = imresize(fused_image,size(image2));
	
	a_original(i,:) = evaluation_original(image1, image2, fused_image);
	a_improved(i,:) = evaluation_improved(text_guided_image, image1, image2, fused_image);
	a_textual(i,:) = a_original(i,:)*(1-Confidence(i)) + a_improved(i,:)*Confidence(i);
end
b(:) = sum(a_textual(:,:))/tot;

disp(b);
writetable(table(b),strcat(fused_path,"/Average_textual_",method,".xlsx"),'WriteVariableNames',false,'Sheet','Sheet1','Range','B2');	

%delete(gcp('nocreate')); for parallel calculation