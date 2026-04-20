use foundations_calculus::{
    gradient_descent, numerical_derivative, numerical_gradient,
    relu, relu_derivative, sigmoid, sigmoid_derivative, tanh_derivative,
};

const EPS: f64 = 1e-4; // 数值误差容忍阈值

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

// --- 数值微分 ---

#[test]
fn test_numerical_derivative_cubic() {
    // f(x) = x³, f'(x) = 3x²
    let f = |x: f64| x * x * x;
    assert!(approx_eq(numerical_derivative(f, 2.0), 12.0, EPS));
    assert!(approx_eq(numerical_derivative(f, 0.0), 0.0, EPS));
    assert!(approx_eq(numerical_derivative(f, -1.0), 3.0, EPS));
}

#[test]
fn test_numerical_derivative_sin() {
    // f(x) = sin(x), f'(x) = cos(x)
    let x = std::f64::consts::PI / 4.0;
    assert!(approx_eq(
        numerical_derivative(f64::sin, x),
        x.cos(),
        EPS
    ));
}

// --- 数值梯度 ---

#[test]
fn test_numerical_gradient_bowl() {
    // f(x,y) = x² + y²，梯度 = [2x, 2y]
    let f = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
    let grad = numerical_gradient(f, &[3.0, -2.0]);
    assert!(approx_eq(grad[0], 6.0, EPS));
    assert!(approx_eq(grad[1], -4.0, EPS));
}

#[test]
fn test_numerical_gradient_at_origin() {
    let f = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
    let grad = numerical_gradient(f, &[0.0, 0.0]);
    assert!(approx_eq(grad[0], 0.0, EPS));
    assert!(approx_eq(grad[1], 0.0, EPS));
}

// --- sigmoid ---

#[test]
fn test_sigmoid_boundary() {
    assert!(approx_eq(sigmoid(0.0), 0.5, EPS));
    // 大正数趋近于 1，大负数趋近于 0
    assert!(sigmoid(100.0) > 0.9999);
    assert!(sigmoid(-100.0) < 0.0001);
}

#[test]
fn test_sigmoid_symmetry() {
    // sigmoid(-x) = 1 - sigmoid(x)
    for &x in &[0.5_f64, 1.0, 2.0] {
        assert!(approx_eq(sigmoid(-x), 1.0 - sigmoid(x), EPS));
    }
}

#[test]
fn test_sigmoid_derivative_matches_numerical() {
    for &x in &[-1.0_f64, 0.0, 1.0, 2.0] {
        assert!(approx_eq(
            sigmoid_derivative(x),
            numerical_derivative(sigmoid, x),
            EPS
        ));
    }
}

// --- relu ---

#[test]
fn test_relu_positive_and_negative() {
    assert!(approx_eq(relu(2.0), 2.0, EPS));
    assert!(approx_eq(relu(-3.0), 0.0, EPS));
    assert!(approx_eq(relu(0.0), 0.0, EPS));
}

#[test]
fn test_relu_derivative() {
    assert!(approx_eq(relu_derivative(1.0), 1.0, EPS));
    assert!(approx_eq(relu_derivative(-1.0), 0.0, EPS));
    // x=0 定义为 0（次梯度选择）
    assert!(approx_eq(relu_derivative(0.0), 0.0, EPS));
}

// --- tanh ---

#[test]
fn test_tanh_derivative_matches_numerical() {
    for &x in &[-1.0_f64, 0.0, 1.0] {
        assert!(approx_eq(
            tanh_derivative(x),
            numerical_derivative(f64::tanh, x),
            EPS
        ));
    }
}

#[test]
fn test_tanh_derivative_at_origin() {
    // tanh'(0) = 1
    assert!(approx_eq(tanh_derivative(0.0), 1.0, EPS));
}

// --- 梯度下降 ---

#[test]
fn test_gradient_descent_converges() {
    let bowl = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
    let result = gradient_descent(bowl, &[3.0, -2.0], 0.1, 100);
    // 100 步后应收敛到接近原点
    assert!(bowl(&result) < 1e-6);
}

#[test]
fn test_gradient_descent_already_at_minimum() {
    let bowl = |v: &[f64]| v[0] * v[0] + v[1] * v[1];
    let result = gradient_descent(bowl, &[0.0, 0.0], 0.1, 10);
    assert!(approx_eq(result[0], 0.0, EPS));
    assert!(approx_eq(result[1], 0.0, EPS));
}
