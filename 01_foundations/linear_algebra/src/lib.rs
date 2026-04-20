use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use synapse_common::{Backend, CpuBackend};

/// 默认 CPU 后端的向量类型别名，日常使用无需写 <B>
pub type Vec_ = Vector<CpuBackend>;
/// 默认 CPU 后端的矩阵类型别名，日常使用无需写 <B>
pub type Mat = Matrix<CpuBackend>;

// ─────────────────────────────────────────
// 运算符宏：生成 T op T / T op &T / &T op T / &T op &T 四种组合
// 核心逻辑在 &T op &T，其余 delegate
// ─────────────────────────────────────────

macro_rules! impl_binop {
    (Vector, $Trait:ident, $method:ident, $AssignTrait:ident, $assign_method:ident) => {
        impl<B: Backend> $Trait<&Vector<B>> for &Vector<B> {
            type Output = Vector<B>;
            fn $method(self, rhs: &Vector<B>) -> Vector<B> {
                self.$method(rhs)
            }
        }
        impl<B: Backend> $Trait<Vector<B>> for Vector<B> {
            type Output = Vector<B>;
            fn $method(self, rhs: Vector<B>) -> Vector<B> { (&self).$method(&rhs) }
        }
        impl<B: Backend> $Trait<&Vector<B>> for Vector<B> {
            type Output = Vector<B>;
            fn $method(self, rhs: &Vector<B>) -> Vector<B> { (&self).$method(rhs) }
        }
        impl<B: Backend> $Trait<Vector<B>> for &Vector<B> {
            type Output = Vector<B>;
            fn $method(self, rhs: Vector<B>) -> Vector<B> { self.$method(&rhs) }
        }
        impl<B: Backend> $AssignTrait<Vector<B>> for Vector<B> {
            fn $assign_method(&mut self, rhs: Vector<B>) { self.$assign_method(&rhs); }
        }
    };
    (Matrix, $Trait:ident, $method:ident, $AssignTrait:ident, $assign_method:ident) => {
        impl<B: Backend> $Trait<&Matrix<B>> for &Matrix<B> {
            type Output = Matrix<B>;
            fn $method(self, rhs: &Matrix<B>) -> Matrix<B> {
                self.$method(rhs)
            }
        }
        impl<B: Backend> $Trait<Matrix<B>> for Matrix<B> {
            type Output = Matrix<B>;
            fn $method(self, rhs: Matrix<B>) -> Matrix<B> { (&self).$method(&rhs) }
        }
        impl<B: Backend> $Trait<&Matrix<B>> for Matrix<B> {
            type Output = Matrix<B>;
            fn $method(self, rhs: &Matrix<B>) -> Matrix<B> { (&self).$method(rhs) }
        }
        impl<B: Backend> $Trait<Matrix<B>> for &Matrix<B> {
            type Output = Matrix<B>;
            fn $method(self, rhs: Matrix<B>) -> Matrix<B> { self.$method(&rhs) }
        }
        impl<B: Backend> $AssignTrait<Matrix<B>> for Matrix<B> {
            fn $assign_method(&mut self, rhs: Matrix<B>) { self.$assign_method(&rhs); }
        }
    };
}

// ─────────────────────────────────────────
// Vector<B>
// ─────────────────────────────────────────

pub struct Vector<B: Backend = CpuBackend> {
    data: Vec<f64>,
    _b: PhantomData<B>,
}

impl<B: Backend> Vector<B> {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data, _b: PhantomData }
    }

    pub fn zeros(n: usize) -> Self {
        Self::new(vec![0.0; n])
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn as_slice(&self) -> &[f64] { &self.data }

    #[must_use]
    pub fn dot(&self, rhs: &Vector<B>) -> f64 {
        assert_eq!(self.len(), rhs.len(), "向量维度不匹配");
        B::dot(&self.data, &rhs.data)
    }

    #[must_use]
    pub fn l2_norm(&self) -> f64 { self.dot(self).sqrt() }

    #[must_use]
    pub fn l1_norm(&self) -> f64 { self.data.iter().map(|x| x.abs()).sum() }

    /// 外积：self ⊗ rhs → (m×n) 矩阵
    #[must_use]
    pub fn outer(&self, rhs: &Vector<B>) -> Matrix<B> {
        let rows = self.len();
        let cols = rhs.len();
        let data = self.data.iter()
            .flat_map(|&a| rhs.data.iter().map(move |&b| a * b))
            .collect();
        Matrix { rows, cols, data, _b: PhantomData }
    }

    // 运算符核心实现，供宏 delegate
    fn add(&self, rhs: &Vector<B>) -> Vector<B> {
        assert_eq!(self.len(), rhs.len(), "向量维度不匹配");
        let mut out = vec![0.0; self.len()];
        B::vadd(&self.data, &rhs.data, &mut out);
        Vector::new(out)
    }

    fn sub(&self, rhs: &Vector<B>) -> Vector<B> {
        assert_eq!(self.len(), rhs.len(), "向量维度不匹配");
        let mut out = vec![0.0; self.len()];
        B::vsub(&self.data, &rhs.data, &mut out);
        Vector::new(out)
    }
}

impl<B: Backend> Clone for Vector<B> {
    fn clone(&self) -> Self { Self::new(self.data.clone()) }
}

impl<B: Backend> PartialEq for Vector<B> {
    fn eq(&self, other: &Self) -> bool { self.data == other.data }
}

impl<B: Backend> fmt::Debug for Vector<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector({:?})", self.data)
    }
}

impl<B: Backend> Default for Vector<B> {
    fn default() -> Self { Self::zeros(0) }
}

impl<B: Backend> From<Vec<f64>> for Vector<B> {
    fn from(data: Vec<f64>) -> Self { Self::new(data) }
}

impl<B: Backend> From<&[f64]> for Vector<B> {
    fn from(s: &[f64]) -> Self { Self::new(s.to_vec()) }
}

impl<B: Backend> Index<usize> for Vector<B> {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 { &self.data[i] }
}

impl<B: Backend> IndexMut<usize> for Vector<B> {
    fn index_mut(&mut self, i: usize) -> &mut f64 { &mut self.data[i] }
}

impl<'a, B: Backend> IntoIterator for &'a Vector<B> {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    fn into_iter(self) -> Self::IntoIter { self.data.iter() }
}

impl<B: Backend> IntoIterator for Vector<B> {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<f64>;
    fn into_iter(self) -> Self::IntoIter { self.data.into_iter() }
}

impl<B: Backend> Neg for Vector<B> {
    type Output = Self;
    fn neg(self) -> Self { Self::new(self.data.iter().map(|x| -x).collect()) }
}

impl<B: Backend> Neg for &Vector<B> {
    type Output = Vector<B>;
    fn neg(self) -> Vector<B> { Vector::new(self.data.iter().map(|x| -x).collect()) }
}

impl<B: Backend> Mul<f64> for &Vector<B> {
    type Output = Vector<B>;
    fn mul(self, s: f64) -> Vector<B> {
        let mut out = vec![0.0; self.len()];
        B::vscale(&self.data, s, &mut out);
        Vector::new(out)
    }
}

impl<B: Backend> Mul<f64> for Vector<B> {
    type Output = Vector<B>;
    fn mul(self, s: f64) -> Vector<B> { &self * s }
}

impl<B: Backend> MulAssign<f64> for Vector<B> {
    fn mul_assign(&mut self, s: f64) {
        self.data.iter_mut().for_each(|x| *x *= s);
    }
}

impl<B: Backend> AddAssign<&Vector<B>> for Vector<B> {
    fn add_assign(&mut self, rhs: &Vector<B>) {
        assert_eq!(self.len(), rhs.len(), "向量维度不匹配");
        self.data.iter_mut().zip(&rhs.data).for_each(|(a, b)| *a += b);
    }
}

impl<B: Backend> SubAssign<&Vector<B>> for Vector<B> {
    fn sub_assign(&mut self, rhs: &Vector<B>) {
        assert_eq!(self.len(), rhs.len(), "向量维度不匹配");
        self.data.iter_mut().zip(&rhs.data).for_each(|(a, b)| *a -= b);
    }
}

impl_binop!(Vector, Add, add, AddAssign, add_assign);
impl_binop!(Vector, Sub, sub, SubAssign, sub_assign);

impl<B: Backend> fmt::Display for Vector<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, x) in self.data.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:.4}", x)?;
        }
        write!(f, "]")
    }
}

// ─────────────────────────────────────────
// Matrix<B>（行优先存储）
// ─────────────────────────────────────────

pub struct Matrix<B: Backend = CpuBackend> {
    rows: usize,
    cols: usize,
    data: Vec<f64>, // data[i*cols + j] = M[i][j]
    _b: PhantomData<B>,
}

impl<B: Backend> Matrix<B> {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "数据长度与矩阵尺寸不符");
        Self { rows, cols, data, _b: PhantomData }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols, vec![0.0; rows * cols])
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        (0..n).for_each(|i| m.data[i * n + i] = 1.0);
        m
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn as_slice(&self) -> &[f64] { &self.data }

    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// 矩阵向量乘：(m×n) · (n,) → (m,)
    #[must_use]
    pub fn mul_vec(&self, v: &Vector<B>) -> Vector<B> {
        assert_eq!(self.cols, v.len(), "矩阵列数与向量长度不匹配");
        let data = (0..self.rows)
            .map(|i| B::dot(&self.data[i * self.cols..(i + 1) * self.cols], &v.data))
            .collect();
        Vector::new(data)
    }

    #[must_use]
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    // 运算符核心实现，供宏 delegate
    fn add(&self, rhs: &Matrix<B>) -> Matrix<B> {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols), "矩阵尺寸不匹配");
        let mut out = vec![0.0; self.data.len()];
        B::vadd(&self.data, &rhs.data, &mut out);
        Matrix::new(self.rows, self.cols, out)
    }

    fn sub(&self, rhs: &Matrix<B>) -> Matrix<B> {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols), "矩阵尺寸不匹配");
        let mut out = vec![0.0; self.data.len()];
        B::vsub(&self.data, &rhs.data, &mut out);
        Matrix::new(self.rows, self.cols, out)
    }

    fn matmul(&self, rhs: &Matrix<B>) -> Matrix<B> {
        assert_eq!(self.cols, rhs.rows, "矩阵乘法维度不匹配");
        let mut out = vec![0.0; self.rows * rhs.cols];
        B::matmul(&self.data, &rhs.data, &mut out, self.rows, self.cols, rhs.cols);
        Matrix::new(self.rows, rhs.cols, out)
    }
}

impl<B: Backend> Clone for Matrix<B> {
    fn clone(&self) -> Self { Self::new(self.rows, self.cols, self.data.clone()) }
}

impl<B: Backend> PartialEq for Matrix<B> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl<B: Backend> fmt::Debug for Matrix<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix({}x{}, {:?})", self.rows, self.cols, self.data)
    }
}

impl<B: Backend> Default for Matrix<B> {
    fn default() -> Self { Self::zeros(0, 0) }
}

impl<B: Backend> From<Vec<Vec<f64>>> for Matrix<B> {
    fn from(rows: Vec<Vec<f64>>) -> Self {
        let nrows = rows.len();
        assert!(nrows > 0, "行数不能为 0");
        let ncols = rows[0].len();
        assert!(rows.iter().all(|r| r.len() == ncols), "每行长度必须相同");
        Self::new(nrows, ncols, rows.into_iter().flatten().collect())
    }
}

impl<B: Backend> Index<(usize, usize)> for Matrix<B> {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.data[r * self.cols + c]
    }
}

impl<B: Backend> IndexMut<(usize, usize)> for Matrix<B> {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }
}

impl<'a, B: Backend> IntoIterator for &'a Matrix<B> {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    fn into_iter(self) -> Self::IntoIter { self.data.iter() }
}

impl<B: Backend> Mul<f64> for &Matrix<B> {
    type Output = Matrix<B>;
    fn mul(self, s: f64) -> Matrix<B> {
        let mut out = vec![0.0; self.data.len()];
        B::vscale(&self.data, s, &mut out);
        Matrix::new(self.rows, self.cols, out)
    }
}

impl<B: Backend> Mul<f64> for Matrix<B> {
    type Output = Matrix<B>;
    fn mul(self, s: f64) -> Matrix<B> { &self * s }
}

impl<B: Backend> MulAssign<f64> for Matrix<B> {
    fn mul_assign(&mut self, s: f64) {
        self.data.iter_mut().for_each(|x| *x *= s);
    }
}

impl<B: Backend> Mul<&Matrix<B>> for &Matrix<B> {
    type Output = Matrix<B>;
    fn mul(self, rhs: &Matrix<B>) -> Matrix<B> { self.matmul(rhs) }
}

impl<B: Backend> Mul<Matrix<B>> for Matrix<B> {
    type Output = Matrix<B>;
    fn mul(self, rhs: Matrix<B>) -> Matrix<B> { (&self).matmul(&rhs) }
}

impl<B: Backend> Mul<&Matrix<B>> for Matrix<B> {
    type Output = Matrix<B>;
    fn mul(self, rhs: &Matrix<B>) -> Matrix<B> { (&self).matmul(rhs) }
}

impl<B: Backend> Mul<Matrix<B>> for &Matrix<B> {
    type Output = Matrix<B>;
    fn mul(self, rhs: Matrix<B>) -> Matrix<B> { self.matmul(&rhs) }
}

impl<B: Backend> AddAssign<&Matrix<B>> for Matrix<B> {
    fn add_assign(&mut self, rhs: &Matrix<B>) {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols), "矩阵尺寸不匹配");
        self.data.iter_mut().zip(&rhs.data).for_each(|(a, b)| *a += b);
    }
}

impl<B: Backend> SubAssign<&Matrix<B>> for Matrix<B> {
    fn sub_assign(&mut self, rhs: &Matrix<B>) {
        assert_eq!((self.rows, self.cols), (rhs.rows, rhs.cols), "矩阵尺寸不匹配");
        self.data.iter_mut().zip(&rhs.data).for_each(|(a, b)| *a -= b);
    }
}

impl_binop!(Matrix, Add, add, AddAssign, add_assign);
impl_binop!(Matrix, Sub, sub, SubAssign, sub_assign);

impl<B: Backend> fmt::Display for Matrix<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                if j > 0 { write!(f, ", ")?; }
                write!(f, "{:8.4}", self[(i, j)])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}
