#pragma once

#include "Solver.h"
#include "EigenTypes.h"

#ifdef USE_PARDISO
#include "PardisoSolver.h"
#endif

#include <iostream>
#include <Eigen/SparseCholesky>

using namespace std;

/**
 * @brief Newton 类，继承自 Solver 类
 * 
 * 该类实现了牛顿法求解器，用于优化问题中的迭代求解。
 * 
 * The Newton class, inheriting from the Solver class.
 * This class implements a Newton solver for iterative optimization problems.
 */
class Newton : public Solver
{
public:
    /**
     * @brief 构造函数
     * 
     * 初始化 Newton 类的实例。
     * 
     * Constructor
     * Initializes an instance of the Newton class.
     */
    Newton();

    /**
     * @brief 执行一次迭代步骤
     * 
     * @return int 返回迭代步骤的状态
     * 
     * Perform a single iteration step.
     * @return int Returns the status of the iteration step.
     */
    int step();

    /**
     * @brief 进行线搜索
     * 
     * 该函数实现了线搜索算法，用于在当前方向上找到合适的步长。
     * 
     * Perform a line search.
     * This function implements a line search algorithm to find an appropriate step size in the current direction.
     */
    void linesearch();

    /**
     * @brief 测试优化过程的进展
     * 
     * @return bool 返回是否有进展
     * 
     * Test the progress of the optimization process.
     * @return bool Returns whether there is progress.
     */
    bool test_progress();

    /**
     * @brief 初始化内部状态
     * 
     * 该函数用于初始化求解器的内部状态。
     * 
     * Initialize internal state.
     * This function initializes the internal state of the solver.
     */
    void internal_init();

    /**
     * @brief 更新外部网格
     * 
     * 该函数用于更新外部网格数据。
     * 
     * Update the external mesh.
     * This function updates the external mesh data.
     */
    void internal_update_external_mesh();

private:
    /**
     * @brief 包装函数，用于避免翻转的线搜索
     * 
     * @param x 输入矩阵
     * @return double 返回线搜索的评估值
     * 
     * Wrapper function for flip-avoiding line search.
     * @param x Input matrix.
     * @return double Returns the evaluation value of the line search.
     */
    double eval_ls(Mat& x);

    /**
     * @brief 将一个 std::vector 乘以一个常数
     * 
     * @param v 输入向量
     * @param s 常数
     * 
     * Multiply a std::vector by a constant.
     * @param v Input vector.
     * @param s Constant.
     */
    void mult(vector<double>& v, double s);

    /**
     * @brief 网格进展的范数
     * 
     * 该变量用于存储网格进展的范数。
     * 
     * Norm of the progress on the mesh.
     * This variable stores the norm of the progress on the mesh.
     */
    double diff_norm;

    /**
     * @brief 求解 Hp = -g 的求解器
     * 
     * 使用 Eigen 库中的 SimplicialLDLT 求解器（已不再使用）。
     * 
     * Solver that computes Hp = -g.
     * Uses the SimplicialLDLT solver from the Eigen library (no longer used).
     */
    Eigen::SimplicialLDLT<SpMat> solver; // not used anymore

#ifdef USE_PARDISO
    /**
     * @brief 使用 Pardiso 求解器
     * 
     * 如果定义了 USE_PARDISO，则使用 Pardiso 求解器。
     * 
     * Use Pardiso solver.
     * If USE_PARDISO is defined, use the Pardiso solver.
     */
    unique_ptr<PardisoSolver<vector<int>, vector<double>>> pardiso = nullptr;
#else
    /**
     * @brief 是否需要初始化
     * 
     * 该变量用于指示求解器是否需要初始化。
     * 
     * Indicates whether initialization is needed.
     * This variable indicates whether the solver needs initialization.
     */
    bool needs_init = true;
#endif

    /**
     * @brief 上一次迭代的时间
     * 
     * 该变量用于存储上一次迭代的时间。
     * 
     * Time of the previous iteration.
     * This variable stores the time of the previous iteration.
     */
    long long int prevTime;
};