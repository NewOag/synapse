use foundations_calculus::{
    gradient_descent, numerical_derivative, numerical_gradient,
    relu, relu_derivative, sigmoid, sigmoid_derivative, tanh_derivative,
};
use log::{debug, info};

fn bowl(v: &[f64]) -> f64 {
    v[0] * v[0] + v[1] * v[1]
}

fn main() {
    synapse_common::init_logger("calculus");

    // 1. 数值微分验证
    info!("=== 数值微分 ===");
    let f = |x: f64| x * x * x; // f(x) = x³, f'(x) = 3x²
    let x = 2.0_f64;
    let numerical = numerical_derivative(f, x);
    let analytical = 3.0 * x * x;
    info!("f(x) = x³  at x=2.0");
    info!("  数值微分: {:.8}", numerical);
    info!("  解析导数: {:.8}", analytical);
    info!("  误差:     {:.2e}", (numerical - analytical).abs());

    // 2. 激活函数及其导数
    info!("\n=== 激活函数 ===");
    info!(
        "{:<6} {:>10} {:>12} {:>10} {:>12} {:>10} {:>12}",
        "x", "sigmoid", "sigmoid'", "relu", "relu'", "tanh", "tanh'"
    );
    for &x in &[-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
        info!(
            "{:<6.1} {:>10.4} {:>12.4} {:>10.4} {:>12.4} {:>10.4} {:>12.4}",
            x,
            sigmoid(x),
            sigmoid_derivative(x),
            relu(x),
            relu_derivative(x),
            x.tanh(),
            tanh_derivative(x),
        );
    }

    // 验证：激活函数导数与数值微分对比
    info!("\n--- 导数验证 (x=1.0) ---");
    let x = 1.0_f64;
    info!(
        "sigmoid': 解析={:.6}  数值={:.6}",
        sigmoid_derivative(x),
        numerical_derivative(sigmoid, x)
    );
    info!(
        "tanh':    解析={:.6}  数值={:.6}",
        tanh_derivative(x),
        numerical_derivative(f64::tanh, x)
    );

    // 3. 梯度（多元函数）
    info!("\n=== 数值梯度 ===");
    let grad = numerical_gradient(bowl, &[3.0, -2.0]);
    info!("∇f(3, -2) = [{:.4}, {:.4}]  (期望 [6.0, -4.0])", grad[0], grad[1]);

    // 4. 梯度下降
    info!("\n=== 梯度下降：最小化 f(x,y) = x² + y² ===");
    let mut x = vec![3.0_f64, -2.0];
    let lr = 0.1;
    for step in 0..=50 {
        if step % 10 == 0 {
            info!(
                "  step {:>3}: x = [{:.4}, {:.4}]  f = {:.6}",
                step, x[0], x[1], bowl(&x)
            );
        }
        debug!("step {}: grad = {:?}", step, numerical_gradient(bowl, &x));
        if step < 50 {
            let grad = numerical_gradient(bowl, &x);
            for i in 0..x.len() {
                x[i] -= lr * grad[i];
            }
        }
    }

    let result = gradient_descent(bowl, &[3.0, -2.0], 0.1, 50);
    info!(
        "最终结果: x = [{:.6}, {:.6}]  f = {:.2e}",
        result[0],
        result[1],
        bowl(&result)
    );
}
