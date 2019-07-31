#include "parametersStereo.h"

using namespace cv;

bool check_image(const cv::Mat &image, std::string name)
{
	if (!image.data)
	{
		std::cerr << name << " data not loaded.\n";
		return false;
	}
	return true;
}


bool check_dimensions(const cv::Mat &img1, const cv::Mat &img2)
{
	if (img1.size() != img2.size())
	{
		std::cerr << "Images' dimensions do not corresponds.";
		return false;
	}
	return true;
}

/**
* @brief convert cv::Point2d(in OpenCV) to cv::Point(in OpenCV)
* @param point
* @return
*/
Point2d PointF2D(Point2f point)
{
	Point2d pointD = Point2d(point.x, point.y);
	return  pointD;
}
/**
 * \brief convert std::vector<cv::Point> to std::vector<cv::Point2d>
 * \param pts
 * \return
 */
std::vector<Point2d> VecPointF2D(std::vector<Point2f> pts)
{
	std::vector<Point2d> ptsD;
	for (std::vector<Point2f>::iterator iter = pts.begin(); iter != pts.end(); ++iter) {
		ptsD.push_back(PointF2D(*iter));
	}
	return ptsD;
}

cv::Mat Vec4iToMat4d(std::vector<std::vector<cv::Mat>> src)
{
	if (src.empty())
	{
		return Mat();
	}
	if (src[0][0].rows != src[0][0].cols)
	{
		return Mat();
	}

	int rows = src.size();
	int cols = src[0].size();
	int winSize = src[0][0].rows;
	int wmat_sizes[] = { rows, cols, winSize, winSize };
	cv::Mat dst(4, wmat_sizes, CV_32F);

#pragma omp parallel for
	for (int cx = 0; cx < cols; ++cx)
		for (int cy = 0; cy < rows; ++cy)		//for each pixel in the image

			for (int x = 0; x < winSize; ++x)
				for (int y = 0; y < winSize; ++y)		//for every neighboring pixel of the anchor pixel
					dst.at<float>(cv::Vec<int, 4> {cy, cx, y, x}) = src[cy][cx].at<float>(y, x);

	return dst;
}
