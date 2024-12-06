#include "Separation.h"
#include "autodiff.h"

#include <omp.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <igl/cat.h>


using namespace std;

Separation::Separation()
{
}

void Separation::init(int n)
{
    // 合并 EVvar1 和 EVvar2 矩阵到 Esep
    // Concatenate EVvar1 and EVvar2 matrices into Esep
    igl::cat(1, EVvar1, EVvar2, Esep);

    // 转置 Esep 矩阵并赋值给 Esept
    // Transpose Esep matrix and assign it to Esept
    Esept = Esep.transpose();

    // 遍历 Esept 矩阵的每一列
    // Iterate over each column of Esept matrix
    for (int i = 0; i < Esept.outerSize(); ++i)
    {
        // 没有内部循环，因为每列只有2个非零值
        // No inner loop because there are only 2 non-zero values per column
        SpMat::InnerIterator it(Esept, i);
        int idx_xi = it.row();
        int idx_xj = (++it).row();
        pair<int, int> p(idx_xi, idx_xj);
        pair2ind.push_back(p);
        ind2pair.emplace(p, i);
        ind2pair.emplace(std::make_pair(p.second, p.first), i);
    }

    // 转置 V2V 矩阵并赋值给 V2Vt
    // Transpose V2V matrix and assign it to V2Vt
    V2Vt = V2V.transpose();

    // 计算 C2C 矩阵
    // Compute C2C matrix
    C2C = V2Vt * V2V;
    SpMat I(V2V.cols(), V2V.cols());
    I.setIdentity();
    C2C -= I;
    C2C.prune(0, 0);

    // 准备 Hessian 矩阵
    // Prepare Hessian matrix
    prepare_hessian(n);

    // 初始化 connect_alphas 和 disconnect_alphas 向量
    // Initialize connect_alphas and disconnect_alphas vectors
    connect_alphas = Vec::Zero(Esep.rows());
    disconnect_alphas = Vec::Ones(Esep.rows());

    // 初始化 edge_lenghts_per_pair 和 no_seam_constraints_per_pair 向量
    // Initialize edge_lenghts_per_pair and no_seam_constraints_per_pair vectors
    edge_lenghts_per_pair = Vec::Ones(Esept.cols());
    no_seam_constraints_per_pair = Vec::Zero(Esept.cols());
}

void Separation::value(const MatX2& X, double& f)
{
    // 计算 EsepP = Esep * X
    // Compute EsepP = Esep * X
    EsepP = Esep * X;

    // 计算每行的平方和
    // Compute the squared sum of each row
    EsepP_squared_rowwise_sum = EsepP.array().pow(2.0).rowwise().sum();

    // 加上 delta
    // Add delta
    EsepP_squared_rowwise_sum_plus_delta = EsepP_squared_rowwise_sum.array() + delta;

    // 根据分离能量类型选择不同的计算方法
    // Choose different calculation methods based on the separation energy type
    switch (sepEType)
    {
        case SeparationEnergy::LOG:
            // 计算 LOG 类型的分离能量
            // Compute LOG type separation energy
            // f = (EsepP_squared_rowwise_sum_plus_delta.array().log() * Lsep).sum();
            f_per_pair = (EsepP_squared_rowwise_sum_plus_delta.array().log() * Lsep);
            break;
        case SeparationEnergy::QUADRATIC:
            // 计算 QUADRATIC 类型的分离能量
            // Compute QUADRATIC type separation energy
            // f = EsepP_squared_rowwise_sum.sum();
            f_per_pair = EsepP_squared_rowwise_sum;
            break;
        case SeparationEnergy::FLAT_LOG:
        {
            // 计算 FLAT_LOG 类型的分离能量
            // Compute FLAT_LOG type separation energy
            // Verified: f = sum over Lsep * log((||xi-xj||^2 / (||xi-xj||^2 + delta)) + 1) (1 xi-xj pair per col (1 & -1) in Esept
            // Store per-pair value for finding the maximal value in current setup
            f_per_pair = Lsep * (EsepP_squared_rowwise_sum.cwiseQuotient(EsepP_squared_rowwise_sum_plus_delta).array() + 1.0).log();
            break;
        }
        case SeparationEnergy::QUOTIENT:
        case SeparationEnergy::QUOTIENT_NEW:
            // 计算 QUOTIENT 和 QUOTIENT_NEW 类型的分离能量
            // Compute QUOTIENT and QUOTIENT_NEW type separation energy
            // 在这段代码中，进行的是逐元素（按元素）除法运算。
            // 具体来说，它计算了 EsepP_squared_rowwise_sum 和 EsepP_squared_rowwise_sum_plus_delta 之间的逐元素商，
            // 并将结果存储在 f_per_pair 中
            f_per_pair = EsepP_squared_rowwise_sum.cwiseQuotient(EsepP_squared_rowwise_sum_plus_delta);
            break;
        default:
            // 未实现的分离能量类型
            // Unimplemented separation energy type
            assert(false && "Unimplemented separation energy");
    }

    // 在考虑绘画之前存储值
    // Store values before taking painting into account
    f_sep_per_pair = f_per_pair;

    // 添加绘画的吸引力
    // Add attraction force from painting
    // alpha * ||xi - xj||^2
    f_per_pair += (connect_alphas + no_seam_constraints_per_pair).cwiseProduct(EsepP_squared_rowwise_sum);

    // 应用绘画的排斥力
    // Apply distraction force from painting
    // f -> alpha * f
    f_per_pair = f_per_pair.cwiseProduct(disconnect_alphas);

    // 打印edge_lenghts_per_pair的形状
    // std::cout << "edge_lenghts_per_pair shape: " << edge_lenghts_per_pair.rows() << " * " << edge_lenghts_per_pair.cols() << std::endl;
    // 添加边长因子
    // Add edge length factor
    f_per_pair = f_per_pair.cwiseProduct(edge_lenghts_per_pair);

    // 如果一对不应成为接缝，它应该有一个高值
    // If a pair shall not be a seam, it should have a high value
    // f_per_pair = f_per_pair.cwiseProduct(no_seam_constraints_per_pair);

    // 将所有值求和
    // Sum everything up
    f = f_per_pair.sum();
}

void Separation::gradient(const MatX2& X, Vec& g)
{
	MatX2 ge;
	switch (sepEType)
	{
		case SeparationEnergy::LOG:
			ge = 2.0 * Esep.transpose() * Lsep * EsepP_squared_rowwise_sum_plus_delta.cwiseInverse().asDiagonal() * EsepP;
			break;
		case SeparationEnergy::QUADRATIC:
			ge = 2.0 * Esep.transpose() * EsepP;
			break;
		case SeparationEnergy::FLAT_LOG:
		{	// Verified: dxi = (2 * delta * (xi - xj)) / ((2*||xi-xj||^2 + delta) * (||xi-xj||^2 + delta)) for xi and dxj = -dxi
			Vec d_vec = Vec::Constant(EsepP_squared_rowwise_sum.rows(), delta);
			Vec two_x_plus_a = (2.0 * EsepP_squared_rowwise_sum) + d_vec;
			Vec d = d_vec.cwiseQuotient(EsepP_squared_rowwise_sum_plus_delta.cwiseProduct(two_x_plus_a));
			ge = 2.0 * Esep.transpose() * d.asDiagonal() * EsepP;
			break;
		}
		case SeparationEnergy::QUOTIENT:
		case SeparationEnergy::QUOTIENT_NEW:
		{
			Vec d_vec = Vec::Constant(EsepP_squared_rowwise_sum.rows(), delta);
			Vec x_plus_d = EsepP_squared_rowwise_sum + d_vec;
			Vec d = d_vec.cwiseQuotient(x_plus_d.cwiseAbs2());
			Vec dconn_e_disconn = (d + connect_alphas + no_seam_constraints_per_pair).cwiseProduct(edge_lenghts_per_pair).cwiseProduct(disconnect_alphas);
			ge = 2.0 * Esept * dconn_e_disconn.asDiagonal() * EsepP;
			break;
		}
		default:
			assert(false && "Unimplemented separation energy");
	}
	g = Eigen::Map<Vec>(ge.data(), 2.0 * ge.rows(), 1);
}

/**
 * @brief 计算分离能量的 Hessian 矩阵
 * 
 * 该函数根据输入的顶点位置矩阵 X，计算分离能量的 Hessian 矩阵。
 * 
 * @param X 输入的顶点位置矩阵，大小为 n x 2，其中 n 是展开的UV图上的三角形汤的顶点的数量，每行表示一个顶点的二维坐标。
 */
void Separation::hessian(const MatX2& X)
{
    int n = X.rows();
    //cout<<"Separation::hessian X.rows() = "<<n<<endl;

    //std::cout << "Esept Matrix shape: " << Esept.rows() << " * " << Esept.cols() << std::endl;

    int threads = omp_get_max_threads();

	// 使用 OpenMP 并行计算 Hessian 矩阵 按列并行计算
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < Esept.outerSize(); ++i)
    { 
		
        // 没有内部循环，因为每列只有2个非零值,因为一条边上有两个顶点
        // No inner loop because there are only 2 non-zero values per column
        int tid = omp_get_thread_num();
        Vec2 xi, xj;
        Mat4 sh;
        int idx_xi, idx_xj, factor;

        // 得到稀疏矩阵 Esept 的第 i 列的迭代器
        SpMat::InnerIterator it(Esept, i);

        // 得到稀疏矩阵 Esept 的第 i 列的第一个非零元素的行索引
        idx_xi = it.row();
        // 得到稀疏矩阵 Esept 的第 i 列的第一个非零元素的值 由于这是个关系矩阵 
        // 要么值为0表示不相连 要么值为1表示相连 取非零元素的话这个值肯定恒为1
        factor = it.value();
        //cout<<"Esept "<<i <<" column factor = "<<factor<<endl;

        idx_xj = (++it).row();
        xi = X.row(idx_xi);
        xj = X.row(idx_xj);
        find_single_hessian(xi, xj, sh);
        //sh *= factor; // 原作者留下的坑，不考虑 factor 也能正常运行，因为 factor 恒为 1

        // 添加额外的因子，如着色和边缘分裂/合并
        // Add the additional factors like coloring and edge splitting/merging
        Mat4 Esep4;
        Esep4 << 1, 0, -1, 0,
                 0, 1, 0, -1,
                -1, 0, 1, 0,
                 0, -1, 0, 1;
        sh += Esep4 * (connect_alphas(i) + no_seam_constraints_per_pair(i));
        sh *= edge_lenghts_per_pair(i);
        sh *= disconnect_alphas(i);
        //sh *= no_seam_constraints_per_pair(i);

        // 将 Hessian 矩阵的上三角部分存储到 SS 向量中
        // Store the upper triangular part of the Hessian matrix into the SS vector
		// | idx_xi     idx_xi+n   idx_xj     idx_xj+n   |
		// |---------------------------------------------|
		// | sh(0,0)    sh(0,1)    sh(0,2)    sh(0,3)    |
		// |            sh(1,1)    sh(1,2)    sh(1,3)    |
		// |                       sh(2,2)    sh(2,3)    |
		// |                                  sh(3,3)    |
        int ind = 10 * i;
        for (int a = 0; a < 4; ++a)
        {
            for (int b = 0; b <= a; ++b)
            {
                SS[ind++] = sh(b, a);
            }
        }
    }
}

void Separation::prepare_hessian(int n)
{
	II.clear();
	JJ.clear();

    // II 矩阵存储 Hessian 矩阵的行索引
    // JJ 矩阵存储 Hessian 矩阵的列索引
	auto PushPair = [&](int i, int j) { II.push_back(i); JJ.push_back(j); };
	for (int i = 0; i < Esept.outerSize(); ++i)
	{
		SpMat::InnerIterator it(Esept, i);
		int idx_xi = it.row();
		int idx_xj = (++it).row();
		// The indices in the small hessians are setup like this:
		// xi, xi+n, xj, xj+n from top to bottom and left to right
		// we traverse only the upper diagonal of each 4x4 hessian
		// and thus store 10 values, gathered in column order.
		// First column
		PushPair(idx_xi,			idx_xi);
		// Second column
		PushPair(idx_xi,			idx_xi + n);
		PushPair(idx_xi + n,	idx_xi + n);
		// Third column
		PushPair(idx_xi,			idx_xj);
		//PushPair(idx_xi + n,	idx_xj);
		PushPair(idx_xj, idx_xi + n);
		PushPair(idx_xj,			idx_xj);
		// Fourth column
		PushPair(idx_xi,			idx_xj + n);
		PushPair(idx_xi + n,	idx_xj + n);
		PushPair(idx_xj,			idx_xj + n);
		PushPair(idx_xj + n,	idx_xj + n);
	}
	SS = vector<double>(II.size(), 0.);
}

// 请见论文附录中的 APPENDIX A SEPARATION GRADIENT AND HESSIAN
void Separation::find_single_hessian(const Vec2& xi, const Vec2& xj, Mat4& h)
{
    // 是否启用加速
    bool speedup = true;

    // 计算 xi 和 xj 之间的差值
    Vec2 dx = xi - xj;

    // 构造一个包含差值和负差值的向量
    Vec4 dxx;
    dxx << dx, -dx;

    // 计算差值的平方和的一半
    double t = 0.5 * dx.squaredNorm();

    // 定义变量以存储一阶和二阶导数
	// fp为 $\hat{s}(t) = \frac{t}{t + \delta}.$ 对于t的偏导
    double fp, fpp;

    // 根据分离能量类型选择不同的计算方法
    switch (sepEType)
    {
        case SeparationEnergy::LOG:
            // LOG 类型的 Hessian 计算（未实现）
            break;
        case SeparationEnergy::QUADRATIC:
            // QUADRATIC 类型的 Hessian 计算（未实现）
            break;
        case SeparationEnergy::FLAT_LOG:
        {
            // 计算 FLAT_LOG 类型的 Hessian 矩阵
            flat_log_single_hessian(xi, xj, h);
            break;
        }
        case SeparationEnergy::QUOTIENT:
        {
            // 计算 QUOTIENT 类型的 Hessian 矩阵
            fp = delta / ((t + delta) * (t + delta));
            fpp = -2 * fp / (t + delta);
            Mat4 Esep4;
            Esep4 << 1, 0, -1, 0,
                     0, 1, 0, -1,
                    -1, 0, 1,  0,
                     0, -1, 0, 1;
            h = fpp * dxx * dxx.transpose() + fp * Esep4;
            h = fp * Esep4;
            break;
        }
        case SeparationEnergy::QUOTIENT_NEW:
        {
            // 计算 QUOTIENT_NEW 类型的 Hessian 矩阵
            fp = delta / ((t + delta) * (t + delta));
            Mat4 Esep4;
            Esep4 << 1, 0, -1, 0,
                     0, 1, 0, -1,
                    -1, 0, 1, 0,
                     0, -1, 0, 1;
            h = fp * Esep4;
            break;
        }
        default:
            // 未实现的分离能量类型
            break;
    }

    // 如果分离能量类型不是 QUOTIENT_NEW，则将 Hessian 矩阵转换为正定矩阵
    if (sepEType != SeparationEnergy::QUOTIENT_NEW)
        make_spd(h);
}

void Separation::flat_log_single_hessian(const Vec2& xi, const Vec2& xj, Mat4& h)
{
    // 提取 xi 和 xj 的坐标分量
    double xi1 = xi(0), xi2 = xi(1);
    double xj1 = xj(0), xj2 = xj(1);

    // 计算 xi 和 xj 之间的差值
    double t4 = xi1 - xj1;
    double t2 = abs(t4);
    double t6 = xi2 - xj2;
    double t3 = abs(t6);

    // 计算差值的平方
    double t5 = t2 * t2;
    double t7 = t3 * t3;

    // 计算 delta 和差值平方和的和
    double t8 = delta + t5 + t7;

    // 计算 1/t8
    double t9 = 1.0 / t8;

    // 计算 t4 的符号
    double t10 = sign(t4);

    // 计算差值平方和
    double t11 = t5 + t7;

    // 计算 1/(t8*t8)
    double t16 = 1.0 / (t8 * t8);

    // 计算中间变量 t22 和 t23
    double t22 = t2 * t9 * t10 * 2.0;
    double t23 = t2 * t10 * t11 * t16 * 2.0;

    // 计算 t12
    double t12 = t22 - t23;

    // 计算 t13 和 t14
    double t13 = t9 * t11;
    double t14 = t13 + 1.0;

    // 计算 t15 和 t17
    double t15 = t10 * t10;
    double t17 = dirac(t4);

    // 计算 1/t14 和 1/(t14*t14)
    double t18 = 1.0 / t14;
    double t21 = 1.0 / (t14 * t14);

    // 计算 t19 和 t20
    double t19 = sign(t6);
    double t20 = 1.0 / (t8 * t8 * t8);

    // 计算 t24 和 t25
    double t24 = t12 * t12;
    double t25 = t9 * t15 * 2.0;

    // 计算 t26 和 t27
    double t26 = t2 * t9 * t17 * 4.0;
    double t27 = t5 * t11 * t15 * t20 * 8.0;

    // 计算 t48, t49 和 t50
    double t48 = t11 * t15 * t16 * 2.0;
    double t49 = t5 * t15 * t16 * 8.0;
    double t50 = t2 * t11 * t16 * t17 * 4.0;

    // 计算 t28
    double t28 = t25 + t26 + t27 - t48 - t49 - t50;

    // 计算 t29
    double t29 = Lsep * t18 * t28;

    // 计算 t30 和 t34
    double t30 = t2 * t3 * t10 * t16 * t19 * 8.0;
    double t34 = t2 * t3 * t10 * t11 * t19 * t20 * 8.0;

    // 计算 t31
    double t31 = t30 - t34;

    // 计算 t32 和 t36
    double t32 = t3 * t9 * t19 * 2.0;
    double t36 = t3 * t11 * t16 * t19 * 2.0;

    // 计算 t33
    double t33 = t32 - t36;

    // 计算 t35 和 t37
    double t35 = Lsep * t18 * t31;
    double t37 = Lsep * t12 * t21 * t33;

    // 计算 t38 和 t39
    double t38 = t19 * t19;
    double t39 = dirac(t6);

    // 计算 t40
    double t40 = t35 + t37;

    // 计算 t41
    double t41 = t33 * t33;

    // 计算 t42, t43 和 t44
    double t42 = t9 * t38 * 2.0;
    double t43 = t3 * t9 * t39 * 4.0;
    double t44 = t7 * t11 * t20 * t38 * 8.0;

    // 计算 t54, t55 和 t56
    double t54 = t11 * t16 * t38 * 2.0;
    double t55 = t7 * t16 * t38 * 8.0;
    double t56 = t3 * t11 * t16 * t39 * 4.0;

    // 计算 t45
    double t45 = t42 + t43 + t44 - t54 - t55 - t56;

    // 计算 t46
    double t46 = Lsep * t18 * t45;

    // 计算 t47
    double t47 = Lsep * t21 * t24;

    // 计算 t51 和 t52
    double t51 = -t29 + t47;
    double t52 = -t35 - t37;

    // 计算 t53
    double t53 = Lsep * t21 * t41;

    // 计算 t57
    double t57 = -t46 + t53;

    // 填充 Hessian 矩阵 h
    h << t29 - Lsep * t21 * t24, t52, t51, t40,
         -Lsep * t18 * t31 - Lsep * t12 * t21 * t33, t46 - Lsep * t21 * t41, t40, t57,
         t51, t40, t29 - t47, t52,
         t40, t57, t52, t46 - t53;
}

inline int Separation::sign(double val)
{
	return (0.0f < val) - (val < 0.0f);
}

inline double Separation::dirac(double val)
{
	return val == 0 ? INF : 0;
}

void Separation::make_spd(Mat4& h)
{
	Eigen::SelfAdjointEigenSolver<Mat4> es(h, Eigen::EigenvaluesOnly);
	Vec4 D = es.eigenvalues();
	double min_ev = D.minCoeff();
	if (min_ev < 0)
		h -= Mat4::Identity()*(min_ev - 1e-6);
}

void Separation::add_to_global_hessian(const Mat4& sh, int idx_xi, int idx_xj, int n, list<Tripletd>& htriplets)
{
    // 按列处理单个面的 Hessian 矩阵 sh
    // Do column by column of single-face hessian sh
	// | idx_xi     idx_xi+n   idx_xj     idx_xj+n   |
	// |---------------------------------------------|
	// | sh(0,0)    sh(0,1)    sh(0,2)    sh(0,3)    |
	// | sh(1,0)    sh(1,1)    sh(1,2)    sh(1,3)    |
	// | sh(2,0)    sh(2,1)    sh(2,2)    sh(2,3)    |
	// | sh(3,0)    sh(3,1)    sh(3,2)    sh(3,3)    |

    // 第一列
    // First column
    htriplets.push_back(Tripletd(idx_xi,            idx_xi,            sh(0, 0)));
    htriplets.push_back(Tripletd(idx_xi + n,        idx_xi,            sh(1, 0)));
    htriplets.push_back(Tripletd(idx_xj,            idx_xi,            sh(2, 0)));
    htriplets.push_back(Tripletd(idx_xj + n,        idx_xi,            sh(3, 0)));

    // 第二列
    // Second column
    htriplets.push_back(Tripletd(idx_xi,            idx_xi + n,        sh(0, 1)));
    htriplets.push_back(Tripletd(idx_xi + n,        idx_xi + n,        sh(1, 1)));
    htriplets.push_back(Tripletd(idx_xj,            idx_xi + n,        sh(2, 1)));
    htriplets.push_back(Tripletd(idx_xj + n,        idx_xi + n,        sh(3, 1)));

    // 第三列
    // Third column
    htriplets.push_back(Tripletd(idx_xi,            idx_xj,            sh(0, 2)));
    htriplets.push_back(Tripletd(idx_xi + n,        idx_xj,            sh(1, 2)));
    htriplets.push_back(Tripletd(idx_xj,            idx_xj,            sh(2, 2)));
    htriplets.push_back(Tripletd(idx_xj + n,        idx_xj,            sh(3, 2)));

    // 第四列
    // Fourth column
    htriplets.push_back(Tripletd(idx_xi,            idx_xj + n,        sh(0, 3)));
    htriplets.push_back(Tripletd(idx_xi + n,        idx_xj + n,        sh(1, 3)));
    htriplets.push_back(Tripletd(idx_xj,            idx_xj + n,        sh(2, 3)));
    htriplets.push_back(Tripletd(idx_xj + n,        idx_xj + n,        sh(3, 3)));
}

void Separation::update_alphas(const Mat& weights, double max_possible)
{
	// factor of strengthening the separation with alpha * ||xi-xj||^2
	connect_alphas = Esep.cwiseAbs()*weights.col(0);

	// the disconnect_alphas do appear as factors for the whole
	// energy, same as edge_lengths_per_pair values
	// so it makes sense to scale them to [min, 1].
	// whereas 1 is the default, i.e. uncolored/uneffected
	// case. for every colored corner, the energy is reduced
	// and thus its min =< disconnect_alpha < 1.
	// the input comes in the range of [0, max], where max <= max_possible
	// we want to map 0 -> 1 and max_possible -> 0
	// since a factor of 0 would free the triangle we only scale
	// to [s, 1] instead, by scaling max_possible up a bit
	disconnect_alphas = 1. - (Esep.cwiseAbs()*weights.col(1) / (2. * 1.1 * max_possible)).array();
}