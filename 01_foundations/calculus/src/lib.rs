/// 中心差分法数值微分
pub fn numerical_derivative(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-5;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// 数值梯度：对多元函数的每个分量求偏导
pub fn numerical_gradient(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let h = 1e-5;
    let mut grad = vec![0.0; x.len()];
    let mut x_mut = x.to_vec();
    for i in 0..x.len() {
        let orig = x_mut[i];
        x_mut[i] = orig + h;
        let fph = f(&x_mut);
        x_mut[i] = orig - h;
        let fmh = f(&x_mut);
        x_mut[i] = orig;
        grad[i] = (fph - fmh) / (2.0 * h);
    }
    grad
}

// --- 激活函数 ---

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn tanh_derivative(x: f64) -> f64 {
    let t = x.tanh();
    1.0 - t * t
}

// --- 梯度下降 ---

/// 梯度下降，返回最终参数
pub fn gradient_descent(
    f: impl Fn(&[f64]) -> f64,
    init: &[f64],
    lr: f64,
    steps: usize,
) -> Vec<f64> {
    let mut x = init.to_vec();
    for _ in 0..steps {
        let grad = numerical_gradient(&f, &x);
        for i in 0..x.len() {
            x[i] -= lr * grad[i];
        }
    }
    x
}
