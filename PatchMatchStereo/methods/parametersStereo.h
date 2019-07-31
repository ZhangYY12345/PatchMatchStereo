#pragma once
#include <opencv2/opencv.hpp>


bool check_image(const cv::Mat &image, std::string name = "Image");
bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2);

cv::Point2d PointF2D(cv::Point2f point);
std::vector<cv::Point2d> VecPointF2D(std::vector<cv::Point2f> pts);
cv::Mat Vec4iToMat4d(std::vector<std::vector<cv::Mat> > src);

template<typename _Tp>
std::vector<_Tp> convertMat2Vector(const cv::Mat mat)
{
	if (mat.isContinuous())
	{
		return (std::vector<_Tp>)(mat.reshape(0, 1));
	}

	cv::Mat mat_ = mat.clone();
	std::vector<_Tp> vecMat = mat_.reshape(0, 1);
	return (std::vector<_Tp>)(mat_.reshape(0, 1));
}