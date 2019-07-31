#pragma once
#ifndef PATCH_MATCH_H
#define PATCH_MATCH_H

#include <opencv2/opencv.hpp>
#include <vector>

//图像特征数据：vector<vector< _ > >
template <typename T>
class Matrix2D
{
public:
	Matrix2D();
	Matrix2D(unsigned int rows, unsigned int cols);
	Matrix2D(unsigned int rows, unsigned int cols, const T& def);

	T& operator()(unsigned int row, unsigned int col);
	const T& operator()(unsigned int row, unsigned int col) const;
	std::vector<std::vector<T> >& getData();

	unsigned int rows;
	unsigned int cols;

private:

	std::vector<std::vector<T>> data;
};




class Plane
{
public:

	Plane();
	Plane(cv::Vec3f point, cv::Vec3f normal);	//平面点法式方程

	float operator[](int idx) const;	//获取系数，idx = 0, 1, 2
	cv::Vec3f operator()();

	cv::Vec3f getPoint();
	cv::Vec3f getNormal();
	cv::Vec3f getCoeff();

	// computing matching point in other view		
	// reparameterized corresopndent plane in other view
	Plane viewTransform(int x, int y, int sign, int& qx, int &qy);

private:

	cv::Vec3f point;	//平面上点(x_0, y_0, z_0)
	cv::Vec3f normal;	//法向量:(A, B, C) --->A*(x - x_0) + B * (y - y_0) + C * (z - z_0) = 0
	cv::Vec3f coeff;	//平面系数：z = a*x + b*y + c   -> a,b,c
};


namespace pm
{
	class PatchMatch
	{
	public:
		PatchMatch(float alpha, float gamma, float tau_c, float tau_g);

		PatchMatch(const PatchMatch &pm) = delete;

		PatchMatch& operator=(const PatchMatch &pm) = delete;

		void operator()(const cv::Mat3b &img1, const cv::Mat3b &img2, int iterations, bool reverse = false);

		void set(const cv::Mat3b &img1, const cv::Mat3b &img2);

		void process(int iterations, bool reverse = false);

		void postProcess();

		cv::Mat1f getLeftDisparityMap() const;

		cv::Mat1f getRightDisparityMap() const;

		float alpha;	//匹配代价计算TAD C+G 算法中，颜色差和梯度差之间的平衡参数
		float gamma;	//匹配代价聚合ASW中自适应权重计算的参数
		float tau_c;	//匹配代价计算TAD C+G 算法中，颜色差绝对值的截断值
		float tau_g;	//匹配代价计算TAD C+G 算法中，梯度差绝对值的截断值

	private:

		float dissimilarity(const cv::Vec3f &pp, const cv::Vec3f &qq, const cv::Vec2f &pg, const cv::Vec2f &qg);

		float plane_match_cost(const Plane &p, int cx, int cy, int ws, int cpv);
		float plane_match_cost_(const Plane &p, int cx, int cy, int ws, int cpv);

		void precompute_pixels_weights(const cv::Mat3b &frame, cv::Mat &weights, int ws);
		void precompute_pixels_weights_(cv::Mat frame, cv::Mat &weights, int winSize);
		void precompute_pixels_weights_2(cv::Mat frame, cv::Mat &weights, int winSize);

		void initialize_random_planes(Matrix2D<Plane> &planes, float max_d);

		void evaluate_planes_cost(int cpv);

		void spatial_propagation(int x, int y, int cpv, int iter);

		void view_propagation(int x, int y, int cpv);

		void plane_refinement(int x, int y, int cpv, float max_delta_z, float max_delta_n, float end_dz);

		void process_pixel(int x, int y, int cpv, int iter);

		void planes_to_disparity(const Matrix2D<Plane> &planes, cv::Mat1f &disp);

		void fill_invalid_pixels(int y, int x, Matrix2D<Plane> &planes, const cv::Mat1b &validity);

		void weighted_median_filter(int cx, int cy, cv::Mat1f &disparity, const cv::Mat &weights, const cv::Mat1b &valid, int ws, bool use_invalid);


		cv::Mat3b views[2];			// left and right view images
		cv::Mat2f grads[2];			// pixels greyscale gradient for both views
		cv::Mat1f disps[2];			// left and right disparity maps

		Matrix2D<Plane> planes[2];	// pixels' planes for left and right view
		cv::Mat1f costs[2];			// planes' costs
		cv::Mat weigs[2];			// precomputed pixels window weights
									//cv::Mat dissm[2];			//pixels dissimilarities, precompute in parallel

		int rows;
		int cols;

	};



	inline cv::Mat1f PatchMatch::getLeftDisparityMap() const
	{
		return this->disps[0];
	}


	inline cv::Mat1f PatchMatch::getRightDisparityMap() const
	{
		return this->disps[1];
	}



	// consider pre-allocated gradients matrix
	void compute_greyscale_gradient(const::cv::Mat3b &frame, cv::Mat2f &gradient);


	inline bool inside(int x, int y, int lbx, int lby, int ubx, int uby)
	{
		return lbx <= x && x < ubx && lby <= y && y < uby;
	}


	inline float disparity(float x, float y, const Plane &p)
	{
		return p[0] * x + p[1] * y + p[2];
	}


	inline float weight(const cv::Vec3f &p, const cv::Vec3f &q, float gamma = 10.0f)
	{
		return std::exp(-cv::norm(p - q, cv::NORM_L1) / gamma);	//norm().NORM_L1:求绝对值和
	}


	template <typename T>
	inline T vecAverage(const T &x, const T &y, float wx)
	{
		return wx * x + (1 - wx) * y;
	}
}

#endif