#include "Energy.h"
#include <iostream>

Energy::Energy()
	:
	separation(make_unique<Separation>()),
	position(make_unique<Position>()),
	symDirichlet(make_unique<DistortionSymDir>()),
	bbox(make_unique<BBox>())
{
}

void Energy::init(unsigned int nf, const MatX2& Vs, const MatX3i& Fs, const MatX3& V, const MatX3i& F)
{
	separation->init(Vs.rows());
	position->init(Fs, Vs.rows());
	symDirichlet->init(V, F, Vs, Fs);
	bbox->init(Vs.rows());
}



void Energy::evaluate_f(const Vec& x, double& f)
{
    // 将输入向量 x 映射到内部变量 X
    map_to_X(x);

    // 定义变量以存储各个能量项的值
    double fs; // Separation 能量项
    double fd; // SymDirichlet 能量项
    double fp; // Position 能量项
    double fb; // BBox 能量项

    // 计算 Separation 能量项
    separation->value(X, fs);

    // 计算 SymDirichlet 能量项
    symDirichlet->value(X, fd);

    // 定义变量以存储 Position 能量项的梯度和 Hessian 矩阵
    Vec gp;
    SpMat hp;

    // 计算 Position 能量项
    position->evaluate_fgh(X, fp, gp, hp, Position::eval_mode::F);

    // 计算 BBox 能量项
    bbox->value(X, fb);

    // 计算总能量 f，使用各个能量项的加权和
    f = (1.0 - lambda) * fd + lambda * fs + pos_weight * fp + bbox_weight * fb;
}

// 计算能量函数的梯度 g 和 Hessian 矩阵 h
void Energy::evaluate_fgh(const Vec& x, double& f, Vec& g)
{
    // 将输入向量 x 映射到内部变量 X
    map_to_X(x);

    // 定义变量以存储各个能量项的值和梯度
    double fs, fd; // Separation 和 SymDirichlet 能量项
    Vec gs, gd;    // Separation 和 SymDirichlet 能量项的梯度
    SpMat hd;      // SymDirichlet 能量项的 Hessian 矩阵

    // 计算 Separation 能量项、梯度和 Hessian 矩阵
    separation->value(X, fs);
    separation->gradient(X, gs);
    separation->hessian(X);

    // 记录 Separation 能量项的最大值和梯度的范数
    max_sep = separation->f_per_pair.maxCoeff();
    grad_norm_sep = gs.norm();

    // 计算 SymDirichlet 能量项、梯度和 Hessian 矩阵
    symDirichlet->value(X, fd);
    symDirichlet->gradient(X, gd);
    symDirichlet->hessian(X);

    // 记录 SymDirichlet 能量项的最大值和梯度的范数
    max_dist = symDirichlet->Efi.maxCoeff();
    grad_norm_dist = gd.norm();

    // 定义变量以存储 Position 能量项、梯度和 Hessian 矩阵
    double fp;
    Vec gp;
    SpMat hp;

    // 计算 Position 能量项、梯度和 Hessian 矩阵
    position->evaluate_fgh(X, fp, gp, hp, Position::eval_mode::FGH);

    // 记录 Position 能量项的最大值和梯度的范数
    max_pos = position->max_el;
    grad_norm_pos = gp.norm();

    // 定义变量以存储 BBox 能量项和梯度
    double fb;
    Vec gb;

    // 计算 BBox 能量项、梯度和 Hessian 矩阵
    bbox->value(X, fb);
    bbox->gradient(X, gb);
    bbox->hessian(X);

    // 记录 BBox 能量项的最大值和梯度的范数
    max_bbox = bbox->f_max;
    grad_norm_bbox = gb.norm();

    // 计算总能量 f，使用各个能量项的加权和
    f = (1.0 - lambda) * fd + lambda * fs + pos_weight * fp + bbox_weight * fb;

    // 打印矩阵gd gs gp gb的形状
    std::cout << "gd shape: " << gd.rows() << " " << gd.cols() << std::endl;
    std::cout << "gs shape: " << gs.rows() << " " << gs.cols() << std::endl;
    std::cout << "gp shape: " << gp.rows() << " " << gp.cols() << std::endl;
    std::cout << "gb shape: " << gb.rows() << " " << gb.cols() << std::endl;


    // 计算总梯度 g，使用各个能量项的梯度的加权和
    g = (1.0 - lambda) * gd + lambda * gs + pos_weight * gp + bbox_weight * gb;
}

inline void Energy::map_to_X(const Vec& x)
{
    // 将输入向量 x 映射为内部变量 X
    // x 是一个向量，包含了所有顶点的坐标
    // X 是一个矩阵，每行表示一个顶点的坐标
    // 这里使用 Eigen 库的 Map 类将 x 的数据映射为一个矩阵
    X = Eigen::Map<const MatX2>(x.data(), x.rows() / 2, 2);
}