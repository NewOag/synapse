/// 计算后端抽象，隔离基本运算与具体实现。
/// 替换后端只需实现此 trait，调用方类型签名不变（默认 = CpuBackend）。
pub trait Backend: Clone + 'static {
    /// out = a + b（逐元素）
    fn vadd(a: &[f64], b: &[f64], out: &mut [f64]);
    /// out = a - b（逐元素）
    fn vsub(a: &[f64], b: &[f64], out: &mut [f64]);
    /// out = v * s（标量乘）
    fn vscale(v: &[f64], s: f64, out: &mut [f64]);
    /// 点积
    fn dot(a: &[f64], b: &[f64]) -> f64;
    /// 矩阵乘法：(m×k) · (k×n) → out(m×n)，行优先存储，out 须预先清零
    fn matmul(a: &[f64], b: &[f64], out: &mut [f64], m: usize, k: usize, n: usize);
}

/// 纯 CPU 标量实现，当前默认后端
#[derive(Clone)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn vadd(a: &[f64], b: &[f64], out: &mut [f64]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        out.iter_mut().zip(a).zip(b).for_each(|((o, a), b)| *o = a + b);
    }

    fn vsub(a: &[f64], b: &[f64], out: &mut [f64]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), out.len());
        out.iter_mut().zip(a).zip(b).for_each(|((o, a), b)| *o = a - b);
    }

    fn vscale(v: &[f64], s: f64, out: &mut [f64]) {
        debug_assert_eq!(v.len(), out.len());
        out.iter_mut().zip(v).for_each(|(o, x)| *o = x * s);
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn matmul(a: &[f64], b: &[f64], out: &mut [f64], m: usize, k: usize, n: usize) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);
        debug_assert_eq!(out.len(), m * n);
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    out[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
    }
}
