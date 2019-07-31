#include "patchmatch.h"
#include <random>
#include "parametersStereo.h"

#define WINDOW_SIZE 35
#define MAX_DISPARITY 60
#define PLANE_PENALTY 120

/*****************************************
 ***********  Matrix2D  ******************
 *****************************************
 */
template <typename T>
Matrix2D<T>::Matrix2D() {}

template <typename T>
Matrix2D<T>::Matrix2D(unsigned int rows, unsigned int cols)
	: rows(rows), cols(cols), data(rows, std::vector<T>(cols)) { }

template <typename T>
Matrix2D<T>::Matrix2D(unsigned int rows, unsigned int cols, const T &def)
	: rows(rows), cols(cols), data(rows, std::vector<T>(cols, def)) { }

template <typename T>
inline T& Matrix2D<T>::operator()(unsigned int row, unsigned int col)
{
	return data[row][col];
}

template <typename T>
inline const T& Matrix2D<T>::operator()(unsigned int row, unsigned int col) const
{
	return data[row][col];
}

template <typename T>
std::vector<std::vector<T>>& Matrix2D<T>::getData()
{
	return data;
}


/*****************************************
 ***************  Plane  *****************
 *****************************************
 */
Plane::Plane() {}


Plane::Plane(cv::Vec3f point, cv::Vec3f normal) : point(point), normal(normal)
{
	float a = -normal[0] / normal[2];
	float b = -normal[1] / normal[2];
	float c = cv::sum(normal.mul(point))[0] / normal[2];
	coeff = cv::Vec3f(a, b, c);
}

inline float Plane::operator[](int idx) const { return coeff[idx]; }

inline cv::Vec3f Plane::operator()() { return coeff; }

inline cv::Vec3f Plane::getPoint() { return point; }

inline cv::Vec3f Plane::getNormal() { return normal; }

inline cv::Vec3f Plane::getCoeff() { return coeff; }

// computing matching point in other view		
// reparameterized corresopndent plane in other view
/**
 * \brief 获取在target image中与reference image中像素(x,y)匹配的点的坐标和视差平面
 * \param x ：reference image中当前点的x坐标
 * \param y ：reference image中当前点的y坐标
 * \param sign :根据视差求target image中匹配点的x坐标
 * \param qx ：target image中匹配点x坐标
 * \param qy ：target image中匹配点y坐标
 * \return 
 */
Plane Plane::viewTransform(int x, int y, int sign, int& qx, int &qy)
{
	float z = coeff[0] * x + coeff[1] * y + coeff[2];	//当前像素的当前视差平面对应的 视差值
	qx = x + sign * z;		//当前像素在另一视图中对应点的x坐标
	qy = y;

	cv::Vec3f p(qx, qy, z);	//在另一视图中的对应点 与当前点有相同的视差值，视差平面的法向量相同（对应点的视差平面平行）
	return Plane(p, this->normal);
}

/************************************************
 *****************  PatchMatch  *****************
 ************************************************
 */

namespace pm
{

	/**
	 * \brief 计算图像灰度图的梯度：x,y方向的梯度
	 * \param frame ：输入彩色图像RGB图像
	 * \param grad ：2通道图像：[0]:x方向梯度；[1]:y方向梯度
	 */
	void compute_greyscale_gradient(const::cv::Mat3b &frame, cv::Mat2f &grad)
	{
		int scale = 1, delta = 0;
		cv::Mat gray, x_grad, y_grad;

		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::Sobel(gray, x_grad, CV_32F, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);//x方向梯度
		cv::Sobel(gray, y_grad, CV_32F, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);//y方向梯度
		x_grad = x_grad / 8.f;
		y_grad = y_grad / 8.f;

		//#pragma omp parallel for（将x,y方向梯度合并成一张二通道图像）
		//for (int y = 0; y < frame.rows; ++y) {
		//	for (int x = 0; x < frame.cols; ++x) {
		//		grad(y, x)[0] = x_grad.at<float>(y, x);
		//		grad(y, x)[1] = y_grad.at<float>(y, x);
		//	}
		//}
		std::vector<cv::Mat> gradImgs;
		gradImgs.push_back(x_grad);
		gradImgs.push_back(y_grad);

		cv::merge(gradImgs, grad);
	}

	/*******************************************************************
	**************************  PatchMatch  ***************************
	*******************************************************************
	*/
	PatchMatch::PatchMatch(float alpha, float gamma, float tau_c, float tau_g)
		: alpha(alpha), gamma(gamma), tau_c(tau_c), tau_g(tau_g) { }

	/**
	 * \brief 匹配代价计算：TAD C+G
	 * \param pp 
	 * \param qq 
	 * \param pg 
	 * \param qg 
	 * \return 
	 */
	float PatchMatch::dissimilarity(const cv::Vec3f &pp, const cv::Vec3f &qq, const cv::Vec2f &pg, const cv::Vec2f &qg)
	{
		float cost_c = cv::norm(pp - qq, cv::NORM_L1);	//颜色差绝对值
		float cost_g = cv::norm(pg - qg, cv::NORM_L1);	//梯度差绝对值
		cost_c = std::min(cost_c, this->tau_c);	//TAD C
		cost_g = std::min(cost_g, this->tau_g);	//TAD G
		return (1 - this->alpha) * cost_c + this->alpha * cost_g;	//匹配代价 TAD C+G
	}


	// 匹配代价聚类，基于倾斜视窗
	//对每个像素聚合匹配代价
	// aggregated matchig cost of a plane for a pixel
	float PatchMatch::plane_match_cost(const Plane &p, int cx, int cy, int ws, int cpv)
	{
		int sign = -1 + 2 * cpv;

		float cost = 0;
		int half = ws / 2;

		const cv::Mat3b& f1 = views[cpv];		//reference image
		const cv::Mat3b& f2 = views[1 - cpv];	//target image
		const cv::Mat2f& g1 = grads[cpv];		//gradient of reference image
		const cv::Mat2f& g2 = grads[1 - cpv];	//gradient of target image
		const cv::Mat& w1 = weigs[cpv];			//weight computed based on the reference image

		for (int x = cx - half; x <= cx + half; ++x)
		{
			for (int y = cy - half; y <= cy + half; ++y)
			{
				if (!inside(x, y, 0, 0, f1.cols, f1.rows))
					continue;

				//computing disparity
				float dsp = disparity(x, y, p);

				if (dsp < 0 || dsp > MAX_DISPARITY)
				{
					cost += PLANE_PENALTY;
				}
				else
				{
					// find matching point in other view
					float match = x + sign * dsp;	//target image中匹配点的x坐标为浮点型数据，通过插值获取其匹配代价
					int x_match = (int)match;	

					float wm = 1 - (match - x_match);

					if (x_match > f1.cols - 2)	//?
						x_match = f1.cols - 2;
					if (x_match < 0)
						x_match = 0;

					// and evaluating its color and gradinet (averaged)
					cv::Vec3b mcolo = vecAverage(f2(y, x_match), f2(y, x_match + 1), wm);	//插值其颜色特征
					cv::Vec2f mgrad = vecAverage(g2(y, x_match), g2(y, x_match + 1), wm);	//插值其梯度特征

					float w = w1.at<float>(cv::Vec<int, 4>{cy, cx, y - cy + half, x - cx + half});	//基于reference image计算的邻域权重
					cost += w * dissimilarity(f1(y, x), mcolo, g1(y, x), mgrad);
				}
			}
		}
		return cost;
	}

	float PatchMatch::plane_match_cost_(const Plane& p, int cx, int cy, int ws, int cpv)
	{
		int sign = -1 + 2 * cpv;

		int half = ws / 2;

		const cv::Mat3b& f1 = views[cpv];		//reference image
		const cv::Mat3b& f2 = views[1 - cpv];	//target image
		const cv::Mat2f& g1 = grads[cpv];		//gradient of reference image
		const cv::Mat2f& g2 = grads[1 - cpv];	//gradient of target image
		const cv::Mat& w1 = weigs[cpv];			//weight computed based on the reference image

		cv::Mat cost_mat = cv::Mat::zeros(ws, ws, CV_32FC1);

#pragma omp parallel for
		for (int x = cx - half; x <= cx + half; ++x)
		{
			for (int y = cy - half; y <= cy + half; ++y)
			{
				if (!inside(x, y, 0, 0, f1.cols, f1.rows))
					continue;

				//computing disparity
				float dsp = disparity(x, y, p);

				if (dsp < 0 || dsp > MAX_DISPARITY)
				{
					cost_mat.at<float>(y - cy + half, x - cx + half) = PLANE_PENALTY;
				}
				else
				{
					// find matching point in other view
					float match = x + sign * dsp;	//target image中匹配点的x坐标为浮点型数据，通过插值获取其匹配代价
					int x_match = (int)match;

					float wm = 1 - (match - x_match);

					if (x_match > f1.cols - 2)	//?
						x_match = f1.cols - 2;
					if (x_match < 0)
						x_match = 0;

					// and evaluating its color and gradinet (averaged)
					cv::Vec3b mcolo = vecAverage(f2(y, x_match), f2(y, x_match + 1), wm);	//插值其颜色特征
					cv::Vec2f mgrad = vecAverage(g2(y, x_match), g2(y, x_match + 1), wm);	//插值其梯度特征

					float w = w1.at<float>(cv::Vec<int, 4>{cy, cx, y - cy + half, x - cx + half});	//基于reference image计算的邻域权重
					cost_mat.at<float>(y - cy + half, x - cx + half) = w * dissimilarity(f1(y, x), mcolo, g1(y, x), mgrad);
				}
			}
		}

		return cv::sum(cost_mat)[0];
	}


	/**
	 * \brief 计算各像素点处邻域像素的权重（基于颜色差异计算权重）
	 * \param frame 
	 * \param weights 
	 * \param ws 
	 */
	void PatchMatch::precompute_pixels_weights(const cv::Mat3b &frame, cv::Mat &weights, int ws)
	{
		int half = ws / 2;

#pragma omp parallel for
		for (int cx = 0; cx < frame.cols; ++cx)
			for (int cy = 0; cy < frame.rows; ++cy)		//for each pixel in the image

				for (int x = cx - half; x <= cx + half; ++x)
					for (int y = cy - half; y <= cy + half; ++y)		//for every neighboring pixel of the anchor pixel
						if (inside(x, y, 0, 0, frame.cols, frame.rows))
							weights.at<float>(cv::Vec<int, 4> {cy, cx, y - cy + half, x - cx + half}) = weight(frame(cy, cx), frame(y, x), this->gamma);
	}

	/**
	 * \brief 计算各像素点处邻域像素的权重（基于颜色差异计算权重）
	 *			权重的计算只涉及一张图像：参考图像
	 * \param frame 
	 * \param weights 
	 * \param gamma_ 
	 * \param winSize 
	 */
	void PatchMatch::precompute_pixels_weights_(cv::Mat frame, cv::Mat &weights, int winSize)
	{
		std::vector<std::vector<cv::Mat> > weights_mat;
		if(winSize % 2 == 0)
		{
			return;
		}
		int halfWin = winSize / 2;

		cv::Mat frame_border;
		cv::copyMakeBorder(frame, frame_border, 
			halfWin, halfWin, halfWin, halfWin, cv::BORDER_REFLECT);

		if(frame.channels() == 3)
		{
			for (int y = 0; y < rows; y++)
			{
				std::vector<cv::Mat> rowWeights;
				for (int x = 0; x < cols; x++)
				{
					cv::Mat winImg = frame_border(cv::Rect(x, y, winSize, winSize));

					std::vector<cv::Mat> winImg_cns;
					split(winImg, winImg_cns);
					cv::Mat winDiff_0, winDiff_1, winDiff_2;
					absdiff(winImg_cns[0], winImg_cns[0].at<uchar>(halfWin, halfWin), winDiff_0);
					absdiff(winImg_cns[1], winImg_cns[1].at<uchar>(halfWin, halfWin), winDiff_1);
					absdiff(winImg_cns[2], winImg_cns[2].at<uchar>(halfWin, halfWin), winDiff_2);
					winDiff_0.convertTo(winDiff_0, CV_32FC1);
					winDiff_1.convertTo(winDiff_1, CV_32FC1);
					winDiff_2.convertTo(winDiff_2, CV_32FC1);

					cv::Mat rangeDiff = (winDiff_0 + winDiff_1 + winDiff_2) / this->gamma * (-1);
					cv::Mat dstWeight;
					exp(rangeDiff, dstWeight);
					rowWeights.push_back(dstWeight);
				}
				weights_mat.push_back(rowWeights);
			}
		}
		else if(frame.channels() == 1)
		{
			for (int y = 0; y < rows; y++)
			{
				std::vector<cv::Mat> rowWeights;
				for (int x = 0; x < cols; x++)
				{
					cv::Mat winImg = frame_border(cv::Rect(x, y, winSize, winSize));
					cv::Mat winDiff;
					cv::absdiff(winImg, winImg.at<uchar>(halfWin, halfWin), winDiff);
					winDiff.convertTo(winDiff, CV_32FC1);
					
					cv::Mat rangeDiff = winDiff / this->gamma * (-1);
					cv::Mat dstWeight;
					exp(rangeDiff, dstWeight);
					rowWeights.push_back(dstWeight);
				}
				weights_mat.push_back(rowWeights);
			}
		}

		weights = Vec4iToMat4d(weights_mat);
	}

	void PatchMatch::precompute_pixels_weights_2(cv::Mat frame, cv::Mat& weights, int winSize)
	{
		if (winSize % 2 == 0)
		{
			return;
		}
		int halfWin = winSize / 2;

		cv::Mat frame_border;
		cv::copyMakeBorder(frame, frame_border,
			halfWin, halfWin, halfWin, halfWin, cv::BORDER_REFLECT);

		Matrix2D<cv::Mat> weights_mat(rows, cols);

		if (frame.channels() == 3)
		{
#pragma omp parallel for
			for (int y = 0; y < rows; y++)
			{
				for (int x = 0; x < cols; x++)
				{
					cv::Mat winImg = frame_border(cv::Rect(x, y, winSize, winSize));

					std::vector<cv::Mat> winImg_cns;
					split(winImg, winImg_cns);
					cv::Mat winDiff_0, winDiff_1, winDiff_2;
					absdiff(winImg_cns[0], winImg_cns[0].at<uchar>(halfWin, halfWin), winDiff_0);
					absdiff(winImg_cns[1], winImg_cns[1].at<uchar>(halfWin, halfWin), winDiff_1);
					absdiff(winImg_cns[2], winImg_cns[2].at<uchar>(halfWin, halfWin), winDiff_2);
					winDiff_0.convertTo(winDiff_0, CV_32FC1);
					winDiff_1.convertTo(winDiff_1, CV_32FC1);
					winDiff_2.convertTo(winDiff_2, CV_32FC1);

					cv::Mat rangeDiff = (winDiff_0 + winDiff_1 + winDiff_2) / this->gamma * (-1);
					cv::Mat dstWeight;
					exp(rangeDiff, dstWeight);
					weights_mat(y, x) = dstWeight;
				}
			}
		}
		else if (frame.channels() == 1)
		{
#pragma omp parallel for
			for (int y = 0; y < rows; y++)
			{
				for (int x = 0; x < cols; x++)
				{
					cv::Mat winImg = frame_border(cv::Rect(x, y, winSize, winSize));
					cv::Mat winDiff;
					cv::absdiff(winImg, winImg.at<uchar>(halfWin, halfWin), winDiff);
					winDiff.convertTo(winDiff, CV_32FC1);

					cv::Mat rangeDiff = winDiff / this->gamma * (-1);
					cv::Mat dstWeight;
					exp(rangeDiff, dstWeight);
					weights_mat(y, x) = dstWeight;
				}
			}
		}

		weights = Vec4iToMat4d(weights_mat.getData());
	}


	void PatchMatch::planes_to_disparity(const Matrix2D<Plane> &planes, cv::Mat1f &disp)
	{
		//cv::Mat1f raw_disp(planes.rows, planes.cols);

#pragma omp parallel for
		for (int x = 0; x < cols; ++x)
			for (int y = 0; y < rows; ++y)
				disp(y, x) = disparity(x, y, planes(y, x));

		//cv::normalize(raw_disp, disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	}



	/**
	 * \brief PatchMatch算法的随机初始化：为每个像素随机分配一个[0, max_d)范围内的float类型的视差值
	 * \param planes ：存储每个像素点的视差平面数据
	 * \param max_d ：最大视差值
	 */
	void PatchMatch::initialize_random_planes(Matrix2D<Plane> &planes, float max_d)
	{
		cv::RNG random_generator;
		const int RAND_HALF = RAND_MAX / 2;

#pragma omp parallel for
		for (int y = 0; y<rows; ++y)
		{
			for (int x = 0; x<cols; ++x)
			{
				//随机生成视差值
				float z = random_generator.uniform(.0f, max_d); // random disparity
				cv::Vec3f point(x, y, z);

				//随机生成视差平面的法向量
				float nx = ((float)std::rand() - RAND_HALF) / RAND_HALF;
				float ny = ((float)std::rand() - RAND_HALF) / RAND_HALF;
				float nz = ((float)std::rand() - RAND_HALF) / RAND_HALF;
				cv::Vec3f normal(nx, ny, nz);
				cv::normalize(normal, normal);

				planes(y, x) = Plane(point, normal);		//平面的点法式公式
			}
		}
	}



	/**
	 * \brief 匹配代价聚合
	 * \param cpv ：参考图像的标识：	以左视图为参考图像：0
	 *									以右视图为参考图像：1
	 */
	void PatchMatch::evaluate_planes_cost(int cpv)
	{
		//#pragma omp parallel for是OpenMP中的一个指令,
		//表示接下来的for循环将被多线程执行，另外每次循环之间不能有关系
#pragma omp parallel for
		//对每个像素聚合匹配代价
		for (int y = 0; y<rows; ++y)
			for (int x = 0; x<cols; ++x)
				costs[cpv](y, x) = plane_match_cost_(planes[cpv](y, x), x, y, WINDOW_SIZE, cpv);
	}



	// search for better plane in the neighbourhood of a pixel
	// if iter is even then the function check the left and upper neighbours
	// if iter is odd then the function check the right and lower neighbours
	void PatchMatch::spatial_propagation(int x, int y, int cpv, int iter)
	{
		//std::cerr<<"START SPATIAL PROP\n";
		int rows = views[cpv].rows;
		int cols = views[cpv].cols;

		std::vector<std::pair<int, int> > offsets;

		if (iter % 2 == 0)	//even iteration
		{
			offsets.push_back(std::make_pair(-1, 0));
			offsets.push_back(std::make_pair(0, -1));
		}
		else				//odd iteration
		{
			offsets.push_back(std::make_pair(+1, 0));
			offsets.push_back(std::make_pair(0, +1));
		}

		int sign = (cpv == 0) ? -1 : 1;

		Plane& old_plane = planes[cpv](y, x);
		float& old_cost = costs[cpv](y, x);

		Plane plane[2];
		float cost_[2];

#pragma omp parallel for
		for (int i = 0; i < offsets.size(); i++)
		{
			std::pair<int, int> ofs = offsets[i];

			int ny = y + ofs.first;		//邻点坐标
			int nx = x + ofs.second;

			if (!inside(nx, ny, 0, 0, cols, rows))
				continue;

			plane[i] = planes[cpv](ny, nx);		//邻点的视差平面
			cost_[i] = plane_match_cost_(plane[i], x, y, WINDOW_SIZE, cpv);
		}

		for (int i = 0; i < 2; i++)
		{
			if (cost_[i] < old_cost)
			{
				old_plane = plane[i];
				old_cost = cost_[i];
			}
		}
	}



	void PatchMatch::view_propagation(int x, int y, int cpv)
	{
		int sign = (cpv == 0) ? -1 : 1;	// -1 processing left, +1 processing right

										// current plane for (x,y) in reference image
		Plane view_plane = planes[cpv](y, x);

		// computing matching point in other view		
		// reparameterized correspondent plane in other view
		int mx, my;
		Plane new_plane = view_plane.viewTransform(x, y, sign, mx, my);

		if (!inside(mx, my, 0, 0, views[0].cols, views[0].rows))
			return;

		// check if this reparameterized plane is better in the other view
		float& old_cost = costs[1 - cpv](my, mx);
		float  new_cost = plane_match_cost_(new_plane, mx, my, WINDOW_SIZE, 1 - cpv);

		if (new_cost < old_cost)
		{
			planes[1 - cpv](my, mx) = new_plane;
			old_cost = new_cost;
		}
	}



	void PatchMatch::plane_refinement(int x, int y, int cpv, float max_delta_z, float max_delta_n, float end_dz)
	{
		int sign = (cpv == 0) ? -1 : 1;	// -1 processing left, +1 processing right

		float max_dz = max_delta_z;
		float max_dn = max_delta_n;

		Plane& old_plane = planes[cpv](y, x);
		float& old_cost = costs[cpv](y, x);

		while (max_dz >= end_dz)
		{
			// Searching a random plane starting from the actual one
			std::random_device rd;
			std::mt19937 gen(rd());

			std::uniform_real_distribution<> rand_z(-max_dz, +max_dz);
			std::uniform_real_distribution<> rand_n(-max_dn, +max_dn);

			float z = old_plane[0] * x + old_plane[1] * y + old_plane[2];
			float delta_z = rand_z(gen);
			cv::Vec3f new_point(x, y, z + delta_z);

			cv::Vec3f n = old_plane.getNormal();
			cv::Vec3f delta_n(rand_n(gen), rand_n(gen), rand_n(gen));
			cv::Vec3f new_normal = n + delta_n;
			cv::normalize(new_normal, new_normal);

			// test the new plane
			Plane new_plane(new_point, new_normal);
			float new_cost = plane_match_cost_(new_plane, x, y, WINDOW_SIZE, cpv);

			if (new_cost < old_cost)
			{
				old_plane = new_plane;
				old_cost = new_cost;
			}

			max_dz /= 2.0f;
			max_dn /= 2.0f;
		}
	}



	/**
	 * \brief 针对单个像素的传播，随机搜索过程
	 * \param x ：像素x坐标
	 * \param y ：像素y坐标
	 * \param cpv ：将左视图/右视图作为reference image的标识符
	 * \param iter ：当前迭代次数
	 */
	void PatchMatch::process_pixel(int x, int y, int cpv, int iter)
	{
		// spatial propagation
		spatial_propagation(x, y, cpv, iter);

		// view propagation
		view_propagation(x, y, cpv);

		// plane refinement
		plane_refinement(x, y, cpv, MAX_DISPARITY / 2, 1.0f, 0.1f);

	}



	void PatchMatch::fill_invalid_pixels(int y, int x, Matrix2D<Plane> &planes, const cv::Mat1b &validity)
	{
		int x_lft = x - 1;
		int x_rgt = x + 1;

		if (x_lft >= 0)
		{
			while (x_lft >= 0 && x_lft < cols && !validity(y, x_lft))
				--x_lft;
		}

		if (x_rgt < cols)
		{
			while (x_rgt < cols && x_rgt >= 0 && !validity(y, x_rgt))
				++x_rgt;
		}

		int best_plane_x = x;

		if (x_lft >= 0 && x_rgt < cols)
		{
			float disp_l = disparity(x, y, planes(y, x_lft));
			float disp_r = disparity(x, y, planes(y, x_rgt));
			best_plane_x = (disp_l < disp_r) ? x_lft : x_rgt;
		}
		else if (x_lft >= 0)
			best_plane_x = x_lft;
		else if (x_rgt < cols)
			best_plane_x = x_rgt;

		planes(y, x) = planes(y, best_plane_x);
	}



	void PatchMatch::weighted_median_filter(int cx, int cy, cv::Mat1f &disparity, const cv::Mat &weights, const cv::Mat1b &valid, int ws, bool use_invalid)
	{
		int half = ws / 2;
		float w_tot = 0;
		float w = 0;

		std::vector<std::pair<float, float> > disps_w;

		for (int x = cx - half; x <= cx + half; ++x)
			for (int y = cy - half; y <= cy + half; ++y)
				if (inside(x, y, 0, 0, cols, rows) && (use_invalid || valid(y, x)))
				{
					cv::Vec<int, 4> w_ids({ cy, cx, y - cy + half, x - cx + half });

					w_tot += weights.at<float>(w_ids);
					disps_w.push_back(std::make_pair(weights.at<float>(w_ids), disparity(y, x)));
				}

		std::sort(disps_w.begin(), disps_w.end());

		float med_w = w_tot / 2.0f;

		for (auto dw = disps_w.begin(); dw < disps_w.end(); ++dw)
		{
			w += dw->first;

			if (w >= med_w)
			{
				if (dw == disps_w.begin())
				{
					disparity(cy, cx) = dw->second;
				}
				else
				{
					disparity(cy, cx) = ((dw - 1)->second + dw->second) / 2.0f;
				}
				//disparity(cy, cx) = dw->second;
			}
		}
	}


	void PatchMatch::operator()(const cv::Mat3b &img1, const cv::Mat3b &img2, int iterations, bool reverse)
	{
		this->set(img1, img2);				//计算权重，匹配代价计算，随机初始化视差平面，基于随机初始化视差平面的匹配代价聚合
		this->process(iterations, reverse);	//处理：传播，随机搜索
		this->postProcess();
	}



	/**
	 * \brief 添加输入图像
	 * \param img1 ：输入左图像
	 * \param img2 ：输入右图像
	 */
	void PatchMatch::set(const cv::Mat3b &img1, const cv::Mat3b &img2)
	{
		this->views[0] = img1;
		this->views[1] = img2;

		this->rows = img1.rows;
		this->cols = img1.cols;

		// pixels neighbours weights//计算权值
		std::cerr << "Precomputing pixels weight...\n";
		int wmat_sizes[] = { rows, cols, WINDOW_SIZE, WINDOW_SIZE };
		this->weigs[0] = cv::Mat(4, wmat_sizes, CV_32F);	//4维图像，相当于vector<vector<vector<vector<> > > >:	内层：vector<vector<> >表示单个像素的邻域像素权值
		this->weigs[1] = cv::Mat(4, wmat_sizes, CV_32F);
		precompute_pixels_weights_(img1, this->weigs[0], WINDOW_SIZE);
		precompute_pixels_weights_(img2, this->weigs[1], WINDOW_SIZE);

		//计算梯度，用于匹配代价计算 TAD C+G
		// greyscale images gradient
		std::cerr << "Evaluating images gradient...\n";
		this->grads[0] = cv::Mat2f(rows, cols);
		this->grads[1] = cv::Mat2f(rows, cols);
		compute_greyscale_gradient(img1, this->grads[0]);
		compute_greyscale_gradient(img2, this->grads[1]);

		// pixels' planes random inizialization//视差平面的随机初始化
		std::cerr << "Precomputing random planes...\n";
		this->planes[0] = Matrix2D<Plane>(rows, cols);
		this->planes[1] = Matrix2D<Plane>(rows, cols);
		this->initialize_random_planes(this->planes[0], MAX_DISPARITY);
		this->initialize_random_planes(this->planes[1], MAX_DISPARITY);

		// initial planes costs evaluation
		//基于视差平面聚合匹配代价，reference image中点p(x,y)在target image中的匹配点坐标为：q(x-(ax+by+c), y)
		std::cerr << "Evaluating initial planes cost...\n";
		this->costs[0] = cv::Mat1f(rows, cols);
		this->costs[1] = cv::Mat1f(rows, cols);
		this->evaluate_planes_cost(0);	//以left image为参考图像进行匹配代价聚合
		this->evaluate_planes_cost(1);	//以right image为参考图像进行匹配代价聚合

		// left and right disparity maps
		this->disps[0] = cv::Mat1f(rows, cols);
		this->disps[1] = cv::Mat1f(rows, cols);
	}




	/**
	 * \brief 视差传播，随机搜索
	 * \param iterations ：迭代次数
	 * \param reverse 
	 */
	void PatchMatch::process(int iterations, bool reverse)
	{
		std::cerr << "Processing left and right views...\n";
		for (int iter = 0 + reverse; iter < iterations + reverse; ++iter)
		{
			bool iter_type = (iter % 2 == 0);
			std::cerr << "Iteration " << iter - reverse + 1 << "/" << iterations - reverse << "\r";

			// PROCESS LEFT AND RIGHT VIEW IN SEQUENCE
			for (int work_view = 0; work_view < 2; ++work_view)
			{
				if (iter_type)	//偶数次迭代，从左上角到右下角，依次处理
				{
					for (int y = 0; y < rows; ++y)
						for (int x = 0; x < cols; ++x)
							process_pixel(x, y, work_view, iter);
				}
				else			//奇数次迭代，从右下角到左上角，依次处理
				{
					for (int y = rows - 1; y >= 0; --y)
						for (int x = cols - 1; x >= 0; --x)
							process_pixel(x, y, work_view, iter);
				}
			}
		}
		std::cerr << std::endl;

		this->planes_to_disparity(this->planes[0], this->disps[0]);
		this->planes_to_disparity(this->planes[1], this->disps[1]);
	}



	void PatchMatch::postProcess()
	{
		std::cerr << "Executing post-processing...\n";

		// checking pixels-plane disparity validity
		cv::Mat1b lft_validity(rows, cols, (unsigned char)false);
		cv::Mat1b rgt_validity(rows, cols, (unsigned char)false);

		// cv::Mat1b ld(rows, cols);
		// cv::Mat1b rd(rows, cols);

#pragma omp parallel for
		for (int y = 0; y < rows; ++y)
		{
			for (int x = 0; x < cols; ++x)
			{
				int x_rgt_match = std::max(0.f, std::min((float)cols, x - disps[0](y, x)));
				lft_validity(y, x) = (std::abs(disps[0](y, x) - disps[1](y, x_rgt_match)) <= 1);

				int x_lft_match = std::max(0.f, std::min((float)rows, x + disps[1](y, x)));
				rgt_validity(y, x) = (std::abs(disps[1](y, x) - disps[0](y, x_lft_match)) <= 1);
			}
		}

		// cv::imwrite("l_inv.png", 255*lft_validity);
		// cv::imwrite("r_inv.png", 255*rgt_validity);

		// fill-in holes related to invalid pixels
#pragma omp parallel for
		for (int y = 0; y < rows; y++)
		{
			for (int x = 0; x < cols; x++)
			{
				if (!lft_validity(y, x))
					fill_invalid_pixels(y, x, planes[0], lft_validity);

				if (!rgt_validity(y, x))
					fill_invalid_pixels(y, x, planes[1], rgt_validity);
			}
		}

		this->planes_to_disparity(this->planes[0], this->disps[0]);
		this->planes_to_disparity(this->planes[1], this->disps[1]);

		// cv::normalize(disps[0], ld, 0, 255, cv::NORM_MINMAX);
		// cv::normalize(disps[1], rd, 0, 255, cv::NORM_MINMAX);
		// cv::imwrite("ld2.png", ld);
		// cv::imwrite("rd2.png", rd);

		// applying weighted median filter to left and right view respectively
#pragma omp parallel for
		for (int x = 0; x<cols; ++x)
		{
			for (int y = 0; y<rows; ++y)
			{
				weighted_median_filter(x, y, disps[0], weigs[0], lft_validity, WINDOW_SIZE, false);
				weighted_median_filter(x, y, disps[1], weigs[1], rgt_validity, WINDOW_SIZE, false);
			}
		}
	}

}




