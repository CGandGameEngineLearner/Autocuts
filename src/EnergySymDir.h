#pragma once

#include "EigenTypes.h"
#include "Utils.h"
#include<functional>
#include <Eigen/Core>
#include <Eigen/Sparse>

class DistortionSymDir
{

public:

	/**************************************************************************************************************************/
	//INITIALIZATION 
	DistortionSymDir();

	void init(const MatX3& V, const MatX3i& F, const MatX2& Vs, const MatX3i& Fs);

	void value(const MatX2& X, double& f);

	void gradient(const MatX2& X, Vec& g);

	void hessian(const MatX2& X);

	//loop implementation
	void prepare_hessian(int n);
	/****************************************************************************************************************************/
	double bound=0;
	Eigen::MatrixX3i F;
	Eigen::MatrixX2d V;

	int numV;
	int numE;
	int numS;
	int numF;

	//Jacobian of the parameterization per face
	Eigen::VectorXd a;
	Eigen::VectorXd b;
	Eigen::VectorXd c;
	Eigen::VectorXd d;
	//Eigen::MatrixXd Juv;		//[a,b,c,d]
	//Eigen::MatrixXd invJuv;	//the order and the signs isn't important because the energy is squared anyway thus will be [a,b,c,d]*1/(ad-bc)
	Eigen::VectorXd detJuv;		//(ad-bc)
	Eigen::VectorXd invdetJuv;	//1/(ad-bc)
	Eigen::SparseMatrix<double> DdetJuv_DUV; //jacobian of the function (detJuv) by UV

	//singular values
	Eigen::MatrixX2d s; //Singular values s[0]>s[1]  shape: number of faces * 2
	Eigen::MatrixX4d v; //Singular vectors shape: number of faces * 4   每行是把2*2的局部V矩阵展开的
	Eigen::MatrixX4d u; //Singular vectors

	//singular values dense derivatives s[0]>s[1] 奇异值的稠密导数，s[0] > s[1] 保存了奇异值关于 UV 的偏导数
	//两个矩阵的形状都为 6*面片数量 每一列为一个面片的导数
	//Dsd[0] 保存了 ∂S/∂u和∂S/∂v Dsd[1] 保存了 ∂s/∂u和∂s/∂v 
	//相当于两个奇异值关于 UV 的偏导数
	Eigen::MatrixXd Dsd[2];

	//SVD methods
	// J shape : 2 *2
	// 局部J矩阵为:
	// [  a , b
	//    c ,  d  ]
	bool updateJ(const MatX2& X);

	// 对J矩阵进行SVD分解
	void UpdateSSVDFunction();

	/**
	 * @brief 计算稠密奇异值分解（SVD）导数
	 *
	 * 该函数计算每个面在两个局部坐标方向上的导数信息，并将结果存储在 Dsd 矩阵中。
	 */
	void ComputeDenseSSVDDerivatives();


	//loop implementation
	inline Eigen::Matrix<double, 6, 6> ComputeFaceConeHessian(const Eigen::Matrix<double,6,1> A1, const Eigen::Matrix<double, 6, 1>& A2, double a1x, double a2x);
	inline Mat6 ComputeConvexConcaveFaceHessian( const Vec6& a1, const Vec6& a2, const Vec6& b1, const Vec6& b2, double aY, double bY, double cY, double dY, const Vec6& dSi, const Vec6& dsi, double gradfS, double gradfs, double HS, double Hs);

	//Energy parts
	//distortion
	Eigen::VectorXd Efi;     //Efi=sum(Ef_dist.^2,2), for data->Efi history

	Eigen::MatrixXi Fuv;                             //F of cut mesh for u and v indices 6XnumF 切割网格的 F 矩阵，用于 u 和 v 索引，大小为 6*面片数量
	Eigen::VectorXd Area;                      // 保存了每个三角形面的面积
	Eigen::Matrix3Xd D1d, D2d;						//dense mesh derivative matrices 稠密网格导数矩阵

	Eigen::SparseMatrix<double> a1, a1t, a2, a2t, b1, b1t, b2, b2t;     //constant matrices for cones calcualtion
	Eigen::MatrixXd a1d, a2d, b1d, b2d;					//dense constant matrices for cones calcualtion

//per face hessians vector
   std::vector<Eigen::Matrix<double,6,6>> Hi;
   // pardiso variables
   vector<int> II, JJ;
   vector<double> SS;
};