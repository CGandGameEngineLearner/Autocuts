#include "Newton.h"

#include <chrono>
#include <igl/flip_avoiding_line_search.h>

Newton::Newton() {}

int Newton::step()
{
	eval_fgh(m_x, f, g, h);

 	SSd = energy->symDirichlet->SS;
 	mult(SSd, 1. - energy->lambda);

	SSs = energy->separation->SS;
	mult(SSs, energy->lambda);

	SSp = energy->position->SS;
	mult(SSp, energy->pos_weight);

	SSb = energy->bbox->SS;
	mult(SSb, energy->bbox_weight);

	SS.clear();
	SS.insert(SS.end(), SSd.begin(), SSd.end());
	SS.insert(SS.end(), SSs.begin(), SSs.end());
	SS.insert(SS.end(), SSp.begin(), SSp.end());
	SS.insert(SS.end(), SSb.begin(), SSb.end());

#ifdef USE_PARDISO
	pardiso->update_a(SS);
	try
	{
		pardiso->factorize();
	}
	catch (runtime_error& err)
	{
		cout << err.what();
		return -1;
	}
	Vec rhs = -g;
	pardiso->solve(rhs, p);
#else
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(II.size());
	int rows = *std::max_element(II.begin(), II.end())+1;
	int cols = *std::max_element(JJ.begin(), JJ.end())+1;
	assert(rows == cols && "Rows == Cols at Newton internal init");
	for(int i=0; i<II.size(); i++)
		tripletList.push_back(T(II[i],JJ[i],SS[i]));
	SpMat mat(rows, cols);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	solver.factorize(mat);
	Vec rhs = -g;
	p = solver.solve(rhs);
#endif
	return 0;
}

bool Newton::test_progress()
{
	return true;//g.norm() > 1e-6 && diff_norm > 1e-6;
}

void Newton::internal_init()
{
#ifdef USE_PARDISO
	bool needs_init = pardiso == nullptr;

	if (needs_init)
	{
		pardiso = make_unique<PardisoSolver<vector<int>, vector<double>>>();
		pardiso->set_type(2, true);
	}
#endif

	eval_fgh(m_x, f, g, h);

	IId = energy->symDirichlet->II;
	JJd = energy->symDirichlet->JJ;
	SSd = energy->symDirichlet->SS;

	IIs = energy->separation->II;
	JJs = energy->separation->JJ;
	SSs = energy->separation->SS;

	IIp = energy->position->II;
	JJp = energy->position->JJ;
	SSp = energy->position->SS;

	IIb = energy->bbox->II;
	JJb = energy->bbox->JJ;
	SSb = energy->bbox->SS;

	if (needs_init)
	{ 
		// find pattern for initialization
		II.insert(II.end(), IId.begin(), IId.end());
		II.insert(II.end(), IIs.begin(), IIs.end());
		II.insert(II.end(), IIp.begin(), IIp.end());
		II.insert(II.end(), IIb.begin(), IIb.end());

		JJ.insert(JJ.end(), JJd.begin(), JJd.end());
		JJ.insert(JJ.end(), JJs.begin(), JJs.end());
		JJ.insert(JJ.end(), JJp.begin(), JJp.end());
		JJ.insert(JJ.end(), JJb.begin(), JJb.end());

		SS.insert(SS.end(), SSd.begin(), SSd.end());
		SS.insert(SS.end(), SSs.begin(), SSs.end());
		SS.insert(SS.end(), SSp.begin(), SSp.end());
		SS.insert(SS.end(), SSb.begin(), SSb.end());
#ifdef USE_PARDISO
		pardiso->set_pattern(II, JJ, SS);
		pardiso->analyze_pattern();
#else
		typedef Eigen::Triplet<double> T;
		std::vector<T> tripletList;
		tripletList.reserve(II.size());
		int rows = *std::max_element(II.begin(), II.end()) + 1;
		int cols = *std::max_element(JJ.begin(), JJ.end()) + 1;
		assert(rows == cols && "Rows == Cols at Newton internal init");
		for(int i=0; i<II.size(); i++)
			tripletList.push_back(T(II[i],JJ[i],SS[i]));
		SpMat mat(rows, cols);
		mat.setFromTriplets(tripletList.begin(), tripletList.end());
		solver.analyzePattern(mat);
		needs_init = false;
#endif
	}
}

void Newton::internal_update_external_mesh()
{
	diff_norm = (ext_x - m_x).norm();
	ext_x = m_x;
}


// Newton 类的 linesearch 成员函数
// 该函数执行线搜索算法，用于在牛顿法的每次迭代中找到合适的步长（alpha）
// 以确保能量函数的足够下降，并避免翻转（flip）问题。
void Newton::linesearch()
{
	// 将 m_x 数据转换为 Eigen 的 MatX2 类型，这里假设 m_x 是一个 2D 点的集合，每行代表一个点的两个坐标
	Mat m_x2 = Eigen::Map<MatX2>(m_x.data(), m_x.rows() / 2, 2);
	
	// 将 p 数据转换为 Eigen 的 MatX2 类型，这里 p 代表牛顿法中计算得到的搜索方向
	Mat p2 = Eigen::Map<const MatX2>(p.data(), p.rows() / 2, 2);
	
	// 计算当前点 m_x2 沿着搜索方向 p2 移动后的点 m_plus_p
	Mat m_plus_p = m_x2 + p2;
	
	// 调用 igl::flip_avoiding_line_search 函数执行线搜索，寻找最佳步长 alpha
	// 这个函数会尝试找到一个步长，使得在不翻转网格的同时，能量函数 Fs 有最大的下降
	// eval_ls 是一个成员函数，用于评估给定步长下的能量函数值
	double alpha = igl::flip_avoiding_line_search(Fs, m_x2, m_plus_p, bind(&Newton::eval_ls, this, placeholders::_1));
	
	// 更新 m_x 为新的点集，使用找到的最佳步长 alpha
	// 这里将 m_x2 数据转换回 Vec 类型，存储更新后的点集
	m_x = Eigen::Map<Vec>(m_x2.data(), m_x2.rows() * m_x2.cols());
}

double Newton::eval_ls(Mat& x)
{
	double f;
	Vec g;
	SpMat h;
	Vec vec_x = Eigen::Map<Vec>(x.data(), x.rows()  * x.cols(), 1);
	eval_f(vec_x, f);
	return f;
}

void Newton::mult(vector<double>& v, double s)
{
	for (int i = 0; i < v.size(); ++i)
		v[i] *= s;
}