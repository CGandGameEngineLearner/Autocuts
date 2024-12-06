#pragma once

#include "Energy.h"
#include "EigenTypes.h"

#include <atomic>
#include <functional>
#include <shared_mutex>
using namespace std;

/**
 * @brief Solver 类
 * 
 * 该类实现了一个通用的求解器框架，用于优化问题中的迭代求解。
 * 
 * The Solver class
 * This class implements a general solver framework for iterative optimization problems.
 */
class Solver
{
public:
    /**
     * @brief 构造函数
     * 
     * 初始化 Solver 类的实例。
     * 
     * Constructor
     * Initializes an instance of the Solver class.
     */
    Solver();

    /**
     * @brief 运行求解器
     * 
     * @return int 返回求解器的状态
     * 
     * Run the solver.
     * @return int Returns the status of the solver.
     */
    int run();

    /**
     * @brief 停止求解器
     * 
     * 停止求解器的运行。
     * 
     * Stop the solver.
     * Stops the solver's execution.
     */
    void stop();

    /**
     * @brief 获取网格
     * 
     * @param X 输出的网格顶点位置矩阵
     * 
     * Get the mesh.
     * @param X Output matrix for the mesh vertex positions.
     */
    void get_mesh(MatX2& X);

    /**
     * @brief 初始化求解器
     * 
     * @param V 输入的顶点位置矩阵
     * @param F 输入的面矩阵
     * @param V_cut 切割后的顶点位置矩阵
     * @param F_cut 切割后的面矩阵
     * @param init 初始化参数
     * @param V_loaded 加载的顶点位置矩阵（可选）
     * @param F_loaded 加载的面矩阵（可选）
     * 
     * Initialize the solver.
     * @param V Input vertex position matrix.
     * @param F Input face matrix.
     * @param V_cut Cut vertex position matrix.
     * @param F_cut Cut face matrix.
     * @param init Initialization parameters.
     * @param V_loaded Loaded vertex position matrix (optional).
     * @param F_loaded Loaded face matrix (optional).
     */
    void init(const MatX3& V, const MatX3i& F, const MatX3& V_cut, const MatX3i& F_cut, Utils::Init init, const MatX2& V_loaded = MatX2(), const MatX3i& F_loaded = MatX3i());

    // 指向能量类的指针
    // Pointer to the energy class
    shared_ptr<Energy> energy;

    // 活动标志
    // Activity flags
    atomic_bool is_running{false};
    atomic_bool progressed{false};

    // 显式求解器在步骤完成后的答案
    // Answer from explicit solver after step done
    int ret;

    // 包装器使用的同步函数
    // Synchronization functions used by the wrapper
    void wait_for_param_slot();
    void release_param_slot();

    // 汤顶点和面索引
    // Soup vertices and face indices
    MatX2 Vs;
    MatX3i Fs;

    // 外部（接口）和内部工作网格
    // External (interface) and internal working mesh
    Vec ext_x, m_x;

    int num_steps = 2147483647;

    bool full_init_done = false;

protected:
    // 给包装器一个优雅交叉的机会
    // Give the wrapper a chance to intersect gracefully
    void give_param_slot();
    // 在步骤完成后更新网格
    // Updating the mesh after a step has been done
    void update_external_mesh();

    // 在步骤中评估的下降方向
    // Descent direction evaluated in step
    Vec p;

    // 用于完整和仅值能量评估的函数指针
    // Function pointers to the full and value-only energy evaluation
    function<void(const Vec&, double&)> eval_f;
    function<void(const Vec&, double&, Vec&)> eval_fgh;

    // 当前能量、梯度和 Hessian
    // Current energy, gradient and hessian
    double f;
    Vec g;
    SpMat h;

    // 同步结构
    // Synchronization structures
    atomic_bool params_ready_to_update{false};
    atomic_bool wait_for_param_update{false};
    atomic_bool a_parameter_was_updated{false};
    atomic_bool halt{false};

    // lbfgs 中需要的互斥锁 - 因此受保护
    // Mutex needed in lbfgs - thus protected
    // Zhongshi 于 2017 年 9 月 11 日更改以适应 c++14
    // Changed by Zhongshi at Sep. 11, 2017 to adapt to c++14
    unique_ptr<std::shared_timed_mutex> mesh_mutex;

    // pardiso 变量
    // pardiso variables
    vector<int> IId, JJd, IIs, JJs, IIp, JJp, IIb, JJb, II, JJ;
    vector<double> SSd, SSs, SSp, SSb, SS;

private:
    /**
     * @brief 执行一次迭代步骤
     * 
     * 纯虚函数，子类需要实现具体的迭代步骤。
     * 
     * Perform a single iteration step.
     * Pure virtual function, subclasses need to implement the specific iteration step.
     */
    virtual int step() = 0;

    /**
     * @brief 进行线搜索
     * 
     * 纯虚函数，子类需要实现具体的线搜索算法。
     * 
     * Perform a line search.
     * Pure virtual function, subclasses need to implement the specific line search algorithm.
     */
    virtual void linesearch() = 0;

    /**
     * @brief 测试优化过程的进展
     * 
     * 纯虚函数，子类需要实现具体的进展测试。
     * 
     * Test the progress of the optimization process.
     * Pure virtual function, subclasses need to implement the specific progress test.
     */
    virtual bool test_progress() = 0;

    /**
     * @brief 初始化内部状态
     * 
     * 纯虚函数，子类需要实现具体的初始化过程。
     * 
     * Initialize internal state.
     * Pure virtual function, subclasses need to implement the specific initialization process.
     */
    virtual void internal_init() = 0;

    /**
     * @brief 更新外部网格
     * 
     * 纯虚函数，子类需要实现具体的网格更新过程。
     * 
     * Update the external mesh.
     * Pure virtual function, subclasses need to implement the specific mesh update process.
     */
    virtual void internal_update_external_mesh() = 0;

    // 互斥锁相关
    // Mutex stuff
    unique_ptr<mutex> param_mutex;
    unique_ptr<condition_variable> param_cv;
};