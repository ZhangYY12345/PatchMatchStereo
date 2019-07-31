#include "methods/parametersStereo.h"
#include "methods/patchmatch.h"

using namespace cv;
using namespace std;

int main()
{
	//image loading
	Size imgSize = Size(640, 360);
	Mat stereoPair_rectified_left = imread("D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\rectify\\pcl-test\\10L_rectify.jpg");
	Mat stereoPair_rectified_right = imread("D:\\studying\\stereo vision\\research code\\data\\2019-07-23\\rectify\\pcl-test\\10R_rectify.jpg");

	// Image loading check
	if (!check_image(stereoPair_rectified_left, "Image left") || !check_image(stereoPair_rectified_right, "Image right"))
		return 1;

	// Image sizes check
	if (!check_dimensions(stereoPair_rectified_left, stereoPair_rectified_right))
		return 1;

	if(stereoPair_rectified_left.size() != imgSize)
	{
		resize(stereoPair_rectified_left, stereoPair_rectified_left, imgSize);
		resize(stereoPair_rectified_right, stereoPair_rectified_right, imgSize);
	}
	// processing images
	//asw weight computation parameter:
	const float gamma = 10.0f;
	//TAD C+G parameters:
	const float alpha = 0.9f;
	const float tau_c = 10.0f;	
	const float tau_g = 2.0f;

	pm::PatchMatch patch_match(alpha, gamma, tau_c, tau_g);
	patch_match.set(stereoPair_rectified_left, stereoPair_rectified_right);
	patch_match.process(3);
	patch_match.postProcess();

	cv::Mat1f disp1 = patch_match.getLeftDisparityMap();
	cv::Mat1f disp2 = patch_match.getRightDisparityMap();


	try
	{
		cv::imwrite("PatchMatch_left_disparity.jpg", disp1);
		cv::imwrite("PatchMatch_right_disparity.jpg", disp2);

		cv::normalize(disp1, disp1, 0, 255, cv::NORM_MINMAX);
		cv::normalize(disp2, disp2, 0, 255, cv::NORM_MINMAX);
		cv::imwrite("PatchMatch_left_disparity_0255.jpg", disp1);
		cv::imwrite("PatchMatch_right_disparity_0255.jpg", disp2);
	}
	catch (std::exception &e)
	{
		std::cerr << "Disparity save error.\n" << e.what();
		return 1;
	}

	return 0;
}