use foundations_linear_algebra::{Mat as Matrix, Vec_ as Vector};

const EPS: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

fn vec_approx_eq(a: &Vector, b: &Vector) -> bool {
    a.len() == b.len() && a.into_iter().zip(b).all(|(x, y)| approx_eq(*x, *y))
}

fn mat_approx_eq(a: &Matrix, b: &Matrix) -> bool {
    a.rows() == b.rows()
        && a.cols() == b.cols()
        && a.into_iter().zip(b).all(|(x, y)| approx_eq(*x, *y))
}

// ─── Vector 构造 ──────────────────────────

#[test]
fn test_vector_from_vec() {
    let v = Vector::from(vec![1.0, 2.0, 3.0]);
    assert_eq!(v.len(), 3);
    assert!(approx_eq(v[0], 1.0));
}

#[test]
fn test_vector_from_slice() {
    let v = Vector::from([1.0, 2.0].as_slice());
    assert_eq!(v.len(), 2);
}

#[test]
fn test_vector_default() {
    let v = Vector::default();
    assert!(v.is_empty());
}

// ─── Vector 运算符 ────────────────────────

#[test]
fn test_vector_add_owned() {
    let a = Vector::from(vec![1.0, 2.0]);
    let b = Vector::from(vec![3.0, 4.0]);
    assert!(vec_approx_eq(&(a + b), &Vector::from(vec![4.0, 6.0])));
}

#[test]
fn test_vector_add_ref() {
    let a = Vector::from(vec![1.0, 2.0]);
    let b = Vector::from(vec![3.0, 4.0]);
    assert!(vec_approx_eq(&(&a + &b), &Vector::from(vec![4.0, 6.0])));
    // a, b 仍可用
    assert_eq!(a.len(), 2);
}

#[test]
fn test_vector_add_assign() {
    let mut a = Vector::from(vec![1.0, 2.0]);
    a += Vector::from(vec![3.0, 4.0]);
    assert!(vec_approx_eq(&a, &Vector::from(vec![4.0, 6.0])));
}

#[test]
fn test_vector_sub() {
    let a = Vector::from(vec![4.0, 5.0, 6.0]);
    let b = Vector::from(vec![1.0, 2.0, 3.0]);
    assert!(vec_approx_eq(&(&a - &b), &Vector::from(vec![3.0, 3.0, 3.0])));
}

#[test]
fn test_vector_neg_owned() {
    let a = Vector::from(vec![1.0, -2.0, 3.0]);
    assert!(vec_approx_eq(&(-a), &Vector::from(vec![-1.0, 2.0, -3.0])));
}

#[test]
fn test_vector_neg_ref() {
    let a = Vector::from(vec![1.0, -2.0]);
    let neg = -&a;
    assert!(vec_approx_eq(&neg, &Vector::from(vec![-1.0, 2.0])));
    assert_eq!(a.len(), 2); // a 未被消费
}

#[test]
fn test_vector_mul_scalar() {
    let v = Vector::from(vec![1.0, 2.0, 3.0]);
    assert!(vec_approx_eq(&(&v * 3.0), &Vector::from(vec![3.0, 6.0, 9.0])));
}

#[test]
fn test_vector_mul_assign_scalar() {
    let mut v = Vector::from(vec![1.0, 2.0, 3.0]);
    v *= 2.0;
    assert!(vec_approx_eq(&v, &Vector::from(vec![2.0, 4.0, 6.0])));
}

// ─── Vector 数学运算 ──────────────────────

#[test]
fn test_vector_dot() {
    let a = Vector::from(vec![1.0, 2.0, 3.0]);
    let b = Vector::from(vec![4.0, 5.0, 6.0]);
    assert!(approx_eq(a.dot(&b), 32.0));
}

#[test]
fn test_vector_dot_orthogonal() {
    let a = Vector::from(vec![1.0, 0.0]);
    let b = Vector::from(vec![0.0, 1.0]);
    assert!(approx_eq(a.dot(&b), 0.0));
}

#[test]
fn test_vector_l2_norm() {
    let v = Vector::from(vec![3.0, 4.0]);
    assert!(approx_eq(v.l2_norm(), 5.0));
}

#[test]
fn test_vector_l1_norm() {
    let v = Vector::from(vec![-1.0, 2.0, -3.0]);
    assert!(approx_eq(v.l1_norm(), 6.0));
}

#[test]
fn test_vector_outer() {
    let u = Vector::from(vec![1.0, 2.0]);
    let v = Vector::from(vec![3.0, 4.0]);
    let expected = Matrix::from(vec![vec![3.0, 4.0], vec![6.0, 8.0]]);
    assert!(mat_approx_eq(&u.outer(&v), &expected));
}

// ─── Vector Index ─────────────────────────

#[test]
fn test_vector_index() {
    let v = Vector::from(vec![10.0, 20.0, 30.0]);
    assert!(approx_eq(v[1], 20.0));
}

#[test]
fn test_vector_index_mut() {
    let mut v = Vector::from(vec![1.0, 2.0, 3.0]);
    v[1] = 99.0;
    assert!(approx_eq(v[1], 99.0));
}

// ─── Vector IntoIterator ──────────────────

#[test]
fn test_vector_iter() {
    let v = Vector::from(vec![1.0, 2.0, 3.0]);
    let sum: f64 = (&v).into_iter().sum();
    assert!(approx_eq(sum, 6.0));
}

// ─── Matrix 构造 ──────────────────────────

#[test]
fn test_matrix_from_vec_vec() {
    let m = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
    assert!(approx_eq(m[(0, 1)], 2.0));
}

#[test]
fn test_matrix_identity() {
    let i = Matrix::identity(3);
    for r in 0..3 {
        for c in 0..3 {
            assert!(approx_eq(i[(r, c)], if r == c { 1.0 } else { 0.0 }));
        }
    }
}

#[test]
fn test_matrix_default() {
    let m = Matrix::default();
    assert_eq!(m.rows(), 0);
    assert_eq!(m.cols(), 0);
}

// ─── Matrix 运算符 ────────────────────────

#[test]
fn test_matrix_add_ref() {
    let a = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Matrix::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let expected = Matrix::from(vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
    assert!(mat_approx_eq(&(&a + &b), &expected));
    assert_eq!(a.rows(), 2); // a 未被消费
}

#[test]
fn test_matrix_add_assign() {
    let mut a = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    a += Matrix::from(vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    assert!(approx_eq(a[(0, 0)], 2.0));
}

#[test]
fn test_matrix_sub() {
    let a = Matrix::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let b = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let expected = Matrix::from(vec![vec![4.0, 4.0], vec![4.0, 4.0]]);
    assert!(mat_approx_eq(&(&a - &b), &expected));
}

#[test]
fn test_matrix_mul_scalar() {
    let m = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let expected = Matrix::from(vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
    assert!(mat_approx_eq(&(&m * 2.0), &expected));
}

#[test]
fn test_matrix_mul_assign_scalar() {
    let mut m = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    m *= 3.0;
    assert!(approx_eq(m[(0, 0)], 3.0));
    assert!(approx_eq(m[(1, 1)], 12.0));
}

#[test]
fn test_matrix_mul_ref() {
    let a = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Matrix::from(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let expected = Matrix::from(vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
    assert!(mat_approx_eq(&(&a * &b), &expected));
    assert_eq!(a.rows(), 2); // a 未被消费
}

#[test]
fn test_matrix_mul_identity() {
    let a = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let i = Matrix::identity(2);
    assert!(mat_approx_eq(&(&a * &i), &a));
}

#[test]
fn test_matrix_mul_non_square() {
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let expected = Matrix::from(vec![vec![58.0, 64.0], vec![139.0, 154.0]]);
    assert!(mat_approx_eq(&(a * b), &expected));
}

// ─── Matrix 数学运算 ──────────────────────

#[test]
fn test_matrix_transpose() {
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let at = a.transpose();
    assert_eq!(at.rows(), 3);
    assert_eq!(at.cols(), 2);
    assert!(approx_eq(at[(0, 0)], 1.0));
    assert!(approx_eq(at[(1, 0)], 2.0));
    assert!(approx_eq(at[(2, 0)], 3.0));
    assert!(approx_eq(at[(0, 1)], 4.0));
}

#[test]
fn test_matrix_transpose_twice() {
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(mat_approx_eq(&a.transpose().transpose(), &a));
}

#[test]
fn test_matrix_mul_vec() {
    let m = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let v = Vector::from(vec![1.0, 1.0]);
    assert!(vec_approx_eq(&m.mul_vec(&v), &Vector::from(vec![3.0, 7.0])));
}

#[test]
fn test_matrix_mul_vec_identity() {
    let v = Vector::from(vec![2.0, 5.0, 3.0]);
    let i = Matrix::identity(3);
    assert!(vec_approx_eq(&i.mul_vec(&v), &v));
}

#[test]
fn test_matrix_frobenius_norm() {
    let m = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert!(approx_eq(m.frobenius_norm(), 30_f64.sqrt()));
}

// ─── Matrix Index ─────────────────────────

#[test]
fn test_matrix_index_mut() {
    let mut m = Matrix::zeros(2, 2);
    m[(0, 1)] = 7.0;
    assert!(approx_eq(m[(0, 1)], 7.0));
}

// ─── 数值稳定性 ───────────────────────────

#[test]
fn test_vector_zero_norm() {
    let v = Vector::zeros(5);
    assert!(approx_eq(v.l2_norm(), 0.0));
    assert!(approx_eq(v.l1_norm(), 0.0));
}

#[test]
fn test_matrix_mul_zero() {
    let m = Matrix::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let z = Matrix::zeros(2, 2);
    assert!(mat_approx_eq(&(&m * &z), &z));
}
