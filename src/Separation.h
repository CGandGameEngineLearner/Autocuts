#pragma once

#include "EigenTypes.h"
#include <list>
#include <vector>
#include <map>
#include <limits>

using namespace std;

#ifndef INF
#define INF numeric_limits<double>::infinity()
#endif

class Separation
{
public:
    // 枚举类，用于表示不同的分离能量类型
    enum class SeparationEnergy { LOG, QUADRATIC, FLAT_LOG, QUOTIENT, QUOTIENT_NEW };

    // 构造函数
    Separation();

    // 初始化函数，接受一个整数参数 n
    void init(int n);

    // 计算分离能量值
    void value(const MatX2& X, double& f);

    // 计算分离能量的梯度
    void gradient(const MatX2& X, Vec& g);

    // 计算分离能量的 Hessian 矩阵
    void hessian(const MatX2& X);

    // 计算单个 Hessian 矩阵
    void find_single_hessian(const Vec2& xi, const Vec2& xj, Mat4& h);

    // 更新 alpha 值
    void update_alphas(const Mat& weights, double max_possible);

    // 稀疏矩阵，用于存储各种变量	
    // Esep矩阵为一个表示边与顶点连接关系的矩阵，其大小为三角形汤的总边数 * 2 * 三角形汤中的顶点总数，如果Esep[i][j] == 1，则表示第i条边与第j个顶点相连
    // 由于每条边上有两个顶点，所以Esep的列数是三角形汤中的顶点总数的两倍，并且不包含原网格体上的开放边
    SpMat EVvar1, EVvar2, Esep, Esept, V2V, V2Vt;
    SpMat C2C; // Corner to corner
    MatX2 EsepP;

    // 分离能量的参数
    double Lsep = 1.0, delta = 1.0;
    SeparationEnergy sepEType = SeparationEnergy::QUOTIENT_NEW;

    // 各对之间的分离能量值
    Vec f_per_pair, f_sep_per_pair;

    // Pardiso 矢量
    vector<int> II, JJ;
    vector<double> SS;

    // 强制这些 UV 顶点更紧密地连接，用于梯度计算
    vector<int> gradient_force_connects;

    // 同样用于函数值，影响 f_per_row 中的正确索引
    vector<int> value_force_connects;

    // 强制因子
    double force_factor = 10.;

    // 网格着色指示的权重
    // 通过对每个角力的因子求和来收集 alpha 值
    Vec connect_alphas;
    // 同样用于断开连接
    Vec disconnect_alphas;

    // 每对之间的边长
    Vec edge_lenghts_per_pair;
    Vec no_seam_constraints_per_pair;

    // 用于存储对索引的映射
    vector<std::pair<int, int>> pair2ind;
    map<std::pair<int, int>, int> ind2pair;

private:
    // 内部变量，用于存储 EsepP 的行和的平方和
    Vec EsepP_squared_rowwise_sum;
    Vec EsepP_squared_rowwise_sum_plus_delta;

    // 计算单个 FLAT_LOG 类型的 Hessian 矩阵
    void flat_log_single_hessian(const Vec2& xi, const Vec2& xj, Mat4& h);

    // 将矩阵转换为正定矩阵
    void make_spd(Mat4& h);

    // 将单个 Hessian 矩阵添加到全局 Hessian 矩阵中
    void add_to_global_hessian(const Mat4& sh, int idx_xi, int idx_xj, int n, list<Tripletd>& htriplets);

    // 返回值的符号
    inline int sign(double val);

    // 计算 Dirac 函数值
    inline double dirac(double val);

    // 准备 Hessian 矩阵
    void prepare_hessian(int n);
};