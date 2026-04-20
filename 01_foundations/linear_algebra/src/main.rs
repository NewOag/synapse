use foundations_linear_algebra::{Mat as Matrix, Vec_, Vec_ as Vector};
use log::info;

fn main() {
    synapse_common::init_logger("linear_algebra");

    // 1. 向量运算
    info!("=== 向量运算 ===");
    let a = Vector::from(vec![1.0, 2.0, 3.0]);
    let b = Vector::from(vec![4.0, 5.0, 6.0]);
    info!("a = {}", a);
    info!("b = {}", b);
    info!("a + b = {}", &a + &b);
    info!("a - b = {}", &a - &b);
    info!("a · b = {:.4}", a.dot(&b));
    info!("‖a‖₂ = {:.4}", a.l2_norm());
    info!("‖a‖₁ = {:.4}", a.l1_norm());
    info!("a * 2 = {}", &a * 2.0);

    // 2. 外积
    info!("\n=== 外积 ===");
    let u = Vec_::from(vec![1.0, 2.0]);
    let v = Vec_::from(vec![3.0, 4.0, 5.0]);
    info!("u ⊗ v =\n{}", u.outer(&v));

    // 3. 矩阵运算
    info!("=== 矩阵运算 ===");
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    info!("A (2×3) =\n{}", a);
    info!("B (3×2) =\n{}", b);
    info!("A · B (2×2) =\n{}", &a * &b);
    info!("Aᵀ (3×2) =\n{}", a.transpose());

    // 4. 矩阵向量乘法
    info!("=== 矩阵向量乘法 ===");
    let m = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let v = Vector::from(vec![1.0, 1.0]);
    info!("M =\n{}", m);
    info!("v = {}", v);
    info!("M·v = {}", m.mul_vec(&v));

    // 5. 单位矩阵
    info!("=== 单位矩阵 I₃ ===\n{}", Matrix::identity(3));

    // 6. Frobenius 范数
    let m = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    info!("‖M‖_F = {:.4}", m.frobenius_norm());
}
