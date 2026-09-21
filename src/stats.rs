//! 删失感知的极大似然拟合引擎。
//!
//! 观测类型（y = log10 t，e = 等效 log10 时间；r = 1 - Σ t_i/10^μ_i）：
//!   exact    : 残差 z = y - μ，         -ln φ(z/σ)/σ
//!   censored : “到时未断”是寿命下界，  -ln Φ̄(-μ/σ) （绝不当作在截止时刻断裂）
//!   damage   : 累积损伤模型启用时纳入；断裂用 r 的正态似然，删失用 Φ̄。
//! 分层炉次偏置：一个参考炉次固定为 0，其余炉次 μ += δ_h，δ 带层级收缩惩罚。

use crate::models::{
    self, mean_log_life, n_base, physical, ModelKind, Scale,
};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct Seg {
    pub temp_k: f64,
    pub stress: f64,
    pub hours: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObsKind {
    Exact,
    Censored,
}

#[derive(Clone, Debug)]
pub struct Obs {
    pub sample_id: String,
    pub heat: String,
    pub kind: ObsKind,
    /// 恒载：(温度K, 应力MPa, log10时间)；变载：segments 非空
    pub temp_k: f64,
    pub stress: f64,
    pub y_time: f64,
    pub segments: Vec<Seg>,
    pub variable: bool,
}

#[derive(Clone, Debug)]
pub struct FixedHeat {
    pub heat: String,
    pub bias: f64,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct FitConfig {
    pub kind: ModelKind,
    pub observations: Vec<Obs>,
    pub heats: Vec<String>,                 // 含参考炉次
    pub reference_heat: String,
    pub fixed_heats: Vec<FixedHeat>,        // 用户指定的固定偏置
    pub shrink_tau: f64,                    // 层级偏置收缩标准差（dex）
    pub allow_temp_k: (f64, f64),
    pub allow_stress: (f64, f64),
}

#[derive(Serialize, serde::Deserialize, Clone, Debug)]
pub struct ParamOut {
    pub name: String,
    pub value: f64,
    pub se: Option<f64>,
    pub fixed: bool,
    pub constraint: String,
}

#[derive(Serialize, serde::Deserialize, Clone, Debug)]
pub struct ResidualOut {
    pub sample_id: String,
    pub heat: String,
    pub variable: bool,
    pub obs: String,        // exact | censored
    pub fitted_log10: f64,
    pub residual_dex: Option<f64>,
    pub nll_contribution: f64,
}

#[derive(Serialize, serde::Deserialize, Clone, Debug)]
pub struct FitResult {
    pub model: String,
    pub converged: bool,
    pub nll: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub sigma_dex: f64,
    pub params: Vec<ParamOut>,
    pub heat_bias: Vec<ParamOut>,
    pub residuals: Vec<ResidualOut>,
    pub included: Vec<String>,
    pub scale: Scale,
    pub internal: Internal,
}

#[derive(Serialize, serde::Deserialize, Clone, Debug)]
pub struct Internal {
    pub raw: Vec<f64>,
    pub cov_raw: Vec<Vec<f64>>,
    pub heat_index: Vec<(String, i32)>, // heat -> raw index (-1 = reference 0)
    pub temp_max_k: f64,
}

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const SQRT_2PI: f64 = 2.5066282746310002;

fn norm_log_pdf(z: f64) -> f64 {
    -0.5 * z * z - SQRT_2PI.ln()
}

/// ln Φ(x)，数值稳健。
fn norm_log_cdf(x: f64) -> f64 {
    if x <= -6.0 {
        // 远尾：ln φ(x) - x²/... 的渐近式
        let x2 = x * x;
        -0.5 * x2 - SQRT_2PI.ln() - (1.0 / x - 1.0 / x.powi(3) + 3.0 / x.powi(5)).ln()
    } else {
        let half_erfc = 0.5 * erfc(-x / SQRT_2);
        half_erfc.max(1e-300).ln()
    }
}

/// Abramowitz & Stegun 7.1.26 误差函数补函数（|误差|<1.5e-7）。
fn erfc(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
            - 0.284496736)
            * t
            + 0.254829592)
            * t
            * (-a * a).exp();
    1.0 - sign * y
}

/// 展开参数：物理模型参数 + 自由炉次偏置 + log σ
struct Workspace<'a> {
    cfg: &'a FitConfig,
    nb: usize,
    nfree_heats: usize,
    scale: Scale,
    temp_max_k: f64,
    free_heat_of: Vec<i32>, // cfg.heats 中每炉的 free 索引（-1 参考/固定）
    fixed_bias: Vec<f64>,
}

impl<'a> Workspace<'a> {
    fn build(cfg: &'a FitConfig) -> Self {
        let nb = n_base(cfg.kind);
        let temps: Vec<f64> = cfg
            .observations
            .iter()
            .map(|o| o.temp_k)
            .chain(cfg.observations.iter().flat_map(|o| o.segments.iter().map(|s| s.temp_k)))
            .collect();
        let stresses: Vec<f64> = cfg
            .observations
            .iter()
            .map(|o| o.stress)
            .chain(cfg.observations.iter().flat_map(|o| o.segments.iter().map(|s| s.stress)))
            .collect();
        let t0 = mean(&temps);
        let s0 = mean(&stresses);
        let td = (mean(&temps.iter().map(|t| (t - t0).powi(2)).collect::<Vec<_>>())).sqrt().max(1.0);
        let sd =
            (mean(&stresses.iter().map(|s| (s - s0).powi(2)).collect::<Vec<_>>())).sqrt().max(1.0);
        let temp_max_k = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut free_heat_of = vec![-1i32; cfg.heats.len()];
        let mut fixed_bias = vec![0.0; cfg.heats.len()];
        for fh in &cfg.fixed_heats {
            if let Some(i) = cfg.heats.iter().position(|h| h == &fh.heat) {
                fixed_bias[i] = fh.bias;
            }
        }
        let mut next = 0;
        for (i, h) in cfg.heats.iter().enumerate() {
            if h == &cfg.reference_heat {
                continue;
            }
            if cfg.fixed_heats.iter().any(|fh| &fh.heat == h) {
                continue;
            }
            free_heat_of[i] = next;
            next += 1;
        }
        Workspace {
            cfg,
            nb,
            nfree_heats: next as usize,
            scale: Scale { t0, td, s0, sd },
            temp_max_k,
            free_heat_of,
            fixed_bias,
        }
    }

    fn ndim(&self) -> usize {
        self.nb + self.nfree_heats + 1
    }

    fn sigma_index(&self) -> usize {
        self.nb + self.nfree_heats
    }

    fn phys_params(&self, raw: &[f64]) -> Vec<f64> {
        physical(self.cfg.kind, &raw[..self.nb], self.temp_max_k)
    }

    fn heat_bias(&self, heat_idx: usize, raw: &[f64]) -> f64 {
        match self.free_heat_of[heat_idx] {
            -1 => self.fixed_bias[heat_idx],
            j => raw[self.nb + j as usize],
        }
    }

}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

/// 统一损伤统计量 g = log10 D：恒载 D = t/10^μ；变载 D = Σ h_i/10^{μ_i}。
/// 断裂样本 g ~ N(0,σ²)；右删失样本仅知 g < 0。
fn obs_g(ws: &Workspace, obs: &Obs, p: &[f64], raw: &[f64]) -> f64 {
    let hi = ws.cfg.heats.iter().position(|h| h == &obs.heat).unwrap();
    let bias = ws.heat_bias(hi, raw);
    if obs.variable {
        let d: f64 = obs
            .segments
            .iter()
            .map(|s| s.hours * 10f64.powf(-mean_log_life(ws.cfg.kind, p, s.temp_k, s.stress, &ws.scale) - bias))
            .sum();
        d.max(1e-12).log10()
    } else {
        obs.y_time - (mean_log_life(ws.cfg.kind, p, obs.temp_k, obs.stress, &ws.scale) + bias)
    }
}

fn neg_penalty(ws: &Workspace, raw: &[f64]) -> f64 {
    let mut v = 0.0;
    for j in 0..ws.nfree_heats {
        let d = raw[ws.nb + j];
        v += 0.5 * (d / ws.cfg.shrink_tau).powi(2);
    }
    v
}

struct PerObs {
    g: f64,
    nll: f64,
}

fn eval_all(ws: &Workspace, raw: &[f64]) -> (f64, Vec<PerObs>) {
    let p = ws.phys_params(raw);
    let log_s = raw[ws.sigma_index()];
    let s = log_s.exp().max(1e-4);
    let mut total = neg_penalty(ws, raw);
    let mut rows = Vec::with_capacity(ws.cfg.observations.len());
    for obs in &ws.cfg.observations {
        let g = obs_g(ws, obs, &p, raw);
        let z = g / s;
        let nll = match obs.kind {
            ObsKind::Exact => -norm_log_pdf(z) + log_s,
            ObsKind::Censored => -norm_log_cdf(-z),
        };
        if !nll.is_finite() {
            return (f64::INFINITY, vec![]);
        }
        total += nll;
        rows.push(PerObs { g, nll });
    }
    (total, rows)
}

// ---------- Nelder-Mead ----------
fn nelder_mead<F: Fn(&[f64]) -> f64>(f: &F, x0: &[f64], max_iter: usize) -> (Vec<f64>, f64, bool) {
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += if v[i].abs() > 0.1 { 0.25 * v[i] } else { 0.5 };
        simplex.push(v);
    }
    let mut fv: Vec<f64> = simplex.iter().map(|x| f(x)).collect();
    let mut converged = false;
    for _ in 0..max_iter {
        let mut order: Vec<usize> = (0..n + 1).collect();
        order.sort_by(|&a, &b| fv[a].partial_cmp(&fv[b]).unwrap_or(std::cmp::Ordering::Equal));
        let (lo, hi, hi2) = (order[0], order[n], order[n - 1]);
        let spread = fv[hi] - fv[lo];
        if spread.abs() < 1e-10 {
            converged = true;
            break;
        }
        let mut centroid = vec![0.0; n];
        for &idx in &order[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j] / n as f64;
            }
        }
        let reflect: Vec<f64> = centroid.iter().zip(&simplex[hi]).map(|(c, h)| 2.0 * c - h).collect();
        let fr = f(&reflect);
        if fv[lo] <= fr && fr < fv[hi2] {
            simplex[hi] = reflect;
            fv[hi] = fr;
            continue;
        }
        if fr < fv[lo] {
            let expand: Vec<f64> = centroid
                .iter()
                .zip(&reflect)
                .map(|(c, r)| 2.0 * r - c)
                .collect();
            let fe = f(&expand);
            if fe < fr {
                simplex[hi] = expand;
                fv[hi] = fe;
            } else {
                simplex[hi] = reflect;
                fv[hi] = fr;
            }
            continue;
        }
        let contract: Vec<f64> = centroid
            .iter()
            .zip(&simplex[hi])
            .map(|(c, h)| 0.5 * h + 0.5 * c)
            .collect();
        let fc = f(&contract);
        if fc < fv[hi] {
            simplex[hi] = contract;
            fv[hi] = fc;
            continue;
        }
        for &idx in &order[1..] {
            for j in 0..n {
                simplex[idx][j] = simplex[lo][j] + 0.5 * (simplex[idx][j] - simplex[lo][j]);
            }
            fv[idx] = f(&simplex[idx]);
        }
    }
    let best = (0..n + 1).min_by(|&a, &b| fv[a].partial_cmp(&fv[b]).unwrap()).unwrap();
    (simplex[best].clone(), fv[best], converged)
}

fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(piv, col);
        b.swap(piv, col);
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    Some((0..n).map(|i| b[i] / a[i][i]).collect())
}

/// 用断裂恒载样本做 OLS 得到合理初值。
fn initial_values(ws: &Workspace) -> Vec<f64> {
    let mut raw = vec![0.0; ws.ndim()];
    raw[..n_base(ws.cfg.kind)].copy_from_slice(&models::initial_raw(ws.cfg.kind));
    raw[ws.sigma_index()] = (0.1_f64).ln();
    let rows: Vec<&Obs> = ws
        .cfg
        .observations
        .iter()
        .filter(|o| !o.variable && o.kind == ObsKind::Exact)
        .collect();
    let p = if !rows.is_empty() {
        match ws.cfg.kind {
            ModelKind::LarsonMiller => {
                // T*(y - C) = b0 + b1*s + b2*y
                let (x, yy): (Vec<Vec<f64>>, Vec<f64>) = rows
                    .iter()
                    .map(|o| {
                        (vec![1.0, o.stress, o.y_time], o.temp_k * (o.y_time + models::LM_C))
                    })
                    .collect();
                ols(&x, &yy)
            }
            ModelKind::Custom => {
                let (x, yy): (Vec<Vec<f64>>, Vec<f64>) = rows
                    .iter()
                    .map(|o| {
                        let sn = (o.stress - ws.scale.s0) / ws.scale.sd;
                        let tn = (o.temp_k - ws.scale.t0) / ws.scale.td;
                        (vec![1.0, sn, sn * sn, tn, tn * tn], o.y_time)
                    })
                    .collect();
                ols(&x, &yy)
            }
        }
    } else {
        None
    };
    if let Some(p) = p {
        match ws.cfg.kind {
            ModelKind::LarsonMiller => {
                raw[0] = p[0];
                raw[1] = p[1];
                let target = (p[2] - ws.temp_max_k - 50.0).max(1e-3);
                raw[2] = target + (1.0 - (-target).exp()).ln();
            }
            ModelKind::Custom => {
                for i in 0..5 {
                    let v = match i {
                        2 | 4 => p[i].min(-1e-3),
                        _ => p[i],
                    };
                    if i == 2 || i == 4 {
                        raw[i] = (-v) + (1.0 - (v).exp()).ln();
                    } else {
                        raw[i] = v;
                    }
                }
            }
        }
    }
    // 炉次偏置初值：各炉断裂残差均值（相对参考炉次）
    let pp = ws.phys_params(&raw);
    let ref_r = resid_ref(ws, &pp);
    for (i, h) in ws.cfg.heats.iter().enumerate() {
        if ws.free_heat_of[i] >= 0 {
            let r = mean(&ws
                .cfg
                .observations
                .iter()
                .filter(|o| !o.variable && o.kind == ObsKind::Exact && o.heat == *h)
                .map(|o| o.y_time - mean_log_life(ws.cfg.kind, &pp, o.temp_k, o.stress, &ws.scale))
                .collect::<Vec<_>>());
            let rr = if ref_r.is_nan() { 0.0 } else { ref_r };
            raw[ws.nb + ws.free_heat_of[i] as usize] = r - rr;
        }
    }
    // σ 初值：断裂样本残差标准差
    let resid: Vec<f64> = ws
        .cfg
        .observations
        .iter()
        .filter(|o| !o.variable && o.kind == ObsKind::Exact)
        .map(|o| {
            let hi = ws.cfg.heats.iter().position(|h| h == &o.heat).unwrap();
            o.y_time
                - mean_log_life(ws.cfg.kind, &pp, o.temp_k, o.stress, &ws.scale)
                - ws.heat_bias(hi, &raw)
        })
        .collect();
    let m = mean(&resid);
    let var = mean(&resid.iter().map(|r| (r - m).powi(2)).collect::<Vec<_>>());
    raw[ws.sigma_index()] = var.max(0.0064).ln() * 0.5;
    raw
}

fn resid_ref(ws: &Workspace, pp: &[f64]) -> f64 {
    let h = &ws.cfg.reference_heat;
    mean(&ws
        .cfg
        .observations
        .iter()
        .filter(|o| !o.variable && o.kind == ObsKind::Exact && o.heat == *h)
        .map(|o| o.y_time - mean_log_life(ws.cfg.kind, pp, o.temp_k, o.stress, &ws.scale))
        .collect::<Vec<_>>())
}

fn ols(x: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let p = x[0].len();
    let n = x.len();
    if n < p + 1 {
        return None;
    }
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for (row, &yy) in x.iter().zip(y) {
        for i in 0..p {
            xty[i] += row[i] * yy;
            for j in 0..p {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    solve_linear(xtx, xty)
}

fn numerical_hessian<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], f0: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut h = vec![vec![0.0; n]; n];
    let eps = 1e-4;
    for i in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += eps;
        xm[i] -= eps;
        let fp = f(&xp);
        let fm = f(&xm);
        h[i][i] = (fp - 2.0 * f0 + fm) / (eps * eps);
        for j in (i + 1)..n {
            let mut xpp = x.to_vec();
            let mut xmm = x.to_vec();
            let mut xpm = x.to_vec();
            let mut xmp = x.to_vec();
            xpp[i] += eps; xpp[j] += eps;
            xmm[i] -= eps; xmm[j] -= eps;
            xpm[i] += eps; xpm[j] -= eps;
            xmp[i] -= eps; xmp[j] += eps;
            let v = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * eps * eps);
            h[i][j] = v;
            h[j][i] = v;
        }
    }
    h
}

fn invert_symmetric(mut a: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        if a[piv][col].abs() < 1e-10 {
            return None;
        }
        a.swap(piv, col);
        inv.swap(piv, col);
        let d = a[col][col];
        for c in 0..n {
            a[col][c] /= d;
            inv[col][c] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let fct = a[r][col];
            for c in 0..n {
                a[r][c] -= fct * a[col][c];
                inv[r][c] -= fct * inv[col][c];
            }
        }
    }
    Some(inv)
}

#[derive(Serialize, serde::Deserialize, Clone, Debug)]
pub struct PredictionOut {
    pub temp_k: f64,
    pub stress_mpa: f64,
    pub mean_log10_h: f64,
    pub median_h: f64,
    pub pi_low_h: f64,
    pub pi_high_h: f64,
    pub pi_level: f64,
    pub inside_envelope: bool,
    pub outside_allowed: bool,
    pub risk: String,
    pub boundary_active: Vec<String>,
    pub normalized_distance: f64,
    pub extrapolation_leverage: f64,
    pub warning: Option<String>,
}

#[derive(Serialize, serde::Deserialize, Clone, Debug)]
pub struct Envelope {
    pub temp_min_k: f64,
    pub temp_max_k: f64,
    pub stress_min_mpa: f64,
    pub stress_max_mpa: f64,
}

pub fn envelope(obs: &[Obs]) -> Envelope {
    let mut t_lo = f64::INFINITY;
    let mut t_hi = f64::NEG_INFINITY;
    let mut s_lo = f64::INFINITY;
    let mut s_hi = f64::NEG_INFINITY;
    for o in obs {
        for (t, s) in std::iter::once((o.temp_k, o.stress)).chain(o.segments.iter().map(|g| (g.temp_k, g.stress))) {
            t_lo = t_lo.min(t);
            t_hi = t_hi.max(t);
            s_lo = s_lo.min(s);
            s_hi = s_hi.max(s);
        }
    }
    Envelope {
        temp_min_k: t_lo,
        temp_max_k: t_hi,
        stress_min_mpa: s_lo,
        stress_max_mpa: s_hi,
    }
}

#[allow(dead_code)]
pub struct FitOutput {
    pub result: FitResult,
    pub ws_scale: Scale,
    pub raw: Vec<f64>,
    pub cov: Vec<Vec<f64>>,
    pub temp_max_k: f64,
    pub heat_index: Vec<(String, i32)>,
    pub n_obs: usize,
}

pub fn fit(cfg: FitConfig) -> Option<FitOutput> {
    if cfg.observations.is_empty() {
        return None;
    }
    let ws = Workspace::build(&cfg);
    let x0 = initial_values(&ws);
    let objective = |x: &[f64]| -> f64 {
        let (v, _) = eval_all(&ws, x);
        if v.is_finite() { v } else { 1e12 }
    };
    let (mut xbest, mut fbest, mut conv) = nelder_mead(&objective, &x0, 4000);
    // 第二起点：从数据尺度的默认值出发，规避 OLS 初值不良
    let mut x1 = models::initial_raw(cfg.kind);
    x1.resize(ws.ndim(), 0.0);
    x1[ws.sigma_index()] = (0.1_f64).ln();
    let (xb2, fb2, conv2) = nelder_mead(&objective, &x1, 4000);
    if fb2 < fbest {
        xbest = xb2;
        fbest = fb2;
        conv = conv2;
    }
    let p = ws.phys_params(&xbest);
    let sigma = xbest[ws.sigma_index()].exp();

    // Hessian（NLL）→ 参数协方差；奇异/非正方向用自适应岭脊处理，
    // 使边界平坦方向表现为“较大 SE”而不是伪精确的 0。
    let hess = numerical_hessian(&|x| objective(x), &xbest, fbest);
    let n = xbest.len();
    let scale: Vec<f64> = (0..n)
        .map(|i| hess[i][i].abs().max(1e-6))
        .collect();
    let cov = {
        let mut ridge = 0.0;
        let mut found = None;
        for _ in 0..12 {
            let mut h = hess.clone();
            for i in 0..n {
                h[i][i] += ridge * scale[i] + 1e-8;
            }
            if let Some(inv) = invert_symmetric(h) {
                if (0..n).all(|i| inv[i][i] > 0.0) {
                    found = Some(inv);
                    break;
                }
            }
            ridge = if ridge == 0.0 { 1e-4 } else { ridge * 10.0 };
        }
        found.unwrap_or_else(|| {
            let mut r = vec![vec![0.0; n]; n];
            for i in 0..n {
                // 岭脊完全失败时：大方差明确标记不可辨识（SE=1000）
                r[i][i] = 1e6;
            }
            r
        })
    };

    let (_, rows) = eval_all(&ws, &xbest);
    let n_obs = cfg.observations.len();
    let n_param = ws.ndim() as f64;
    let aic = 2.0 * fbest + 2.0 * n_param;
    let bic = 2.0 * fbest + n_param * (n_obs as f64).ln();

    // 参数输出
    let meta = models::param_meta(cfg.kind);
    let mut params = Vec::new();
    if cfg.kind == ModelKind::LarsonMiller {
        params.push(ParamOut {
            name: "C".into(), value: models::LM_C, se: None, fixed: true,
            constraint: meta[0].1.clone(),
        });
    }
    let base_offset = if cfg.kind == ModelKind::LarsonMiller { 1 } else { 0 };
    for k in 0..ws.nb {
        let constraint = meta[base_offset + k].1.clone();
        params.push(ParamOut {
            name: meta[base_offset + k].0.clone(),
            value: p[k],
            se: cov[k][k].sqrt().into(),
            fixed: false,
            constraint,
        });
    }

    let mut heat_bias = Vec::new();
    let mut heat_index = Vec::new();
    for (i, h) in cfg.heats.iter().enumerate() {
        let idx = ws.free_heat_of[i];
        heat_index.push((h.clone(), idx));
        if h == &cfg.reference_heat {
            heat_bias.push(ParamOut {
                name: h.clone(), value: 0.0, se: None, fixed: true,
                constraint: "参考炉次，偏置固定为 0".into(),
            });
        } else if idx >= 0 {
            let gi = ws.nb + idx as usize;
            heat_bias.push(ParamOut {
                name: h.clone(),
                value: xbest[gi],
                se: cov[gi][gi].sqrt().into(),
                fixed: false,
                constraint: format!("层级收缩 τ={:.3} dex", cfg.shrink_tau),
            });
        } else {
            heat_bias.push(ParamOut {
                name: h.clone(),
                value: ws.fixed_bias[i],
                se: None,
                fixed: true,
                constraint: "用户指定固定偏置".into(),
            });
        }
    }

    let included: Vec<String> = cfg.observations.iter().map(|o| o.sample_id.clone()).collect();
    let residuals = cfg
        .observations
        .iter()
        .zip(&rows)
        .map(|(o, r)| ResidualOut {
            sample_id: o.sample_id.clone(),
            heat: o.heat.clone(),
            variable: o.variable,
            obs: if o.kind == ObsKind::Exact { "exact" } else { "censored" }.into(),
            fitted_log10: if o.variable {
                0.0
            } else {
                let hi = cfg.heats.iter().position(|h| h == &o.heat).unwrap();
                mean_log_life(cfg.kind, &p, o.temp_k, o.stress, &ws.scale)
                    + ws.heat_bias(hi, &xbest)
            },
            residual_dex: if o.variable {
                None
            } else if o.kind == ObsKind::Exact {
                Some(r.g)
            } else {
                None
            },
            nll_contribution: r.nll,
        })
        .collect();

    let result = FitResult {
        model: match cfg.kind {
            ModelKind::LarsonMiller => "larson_miller".into(),
            ModelKind::Custom => "custom".into(),
        },
        converged: conv,
        nll: fbest,
        log_likelihood: -fbest,
        aic,
        bic,
        sigma_dex: sigma,
        params,
        heat_bias,
        residuals,
        included,
        scale: ws.scale,
        internal: Internal {
            raw: xbest.clone(),
            cov_raw: cov.clone(),
            heat_index,
            temp_max_k: ws.temp_max_k,
        },
    };
    Some(FitOutput {
        result,
        ws_scale: ws.scale,
        raw: xbest,
        cov,
        temp_max_k: ws.temp_max_k,
        heat_index: vec![],
        n_obs,
    })
}

impl FitOutput {
    pub fn predict(
        &self,
        env: &Envelope,
        kind: ModelKind,
        temp_k: f64,
        stress: f64,
        allow_temp_k: (f64, f64),
        allow_stress: (f64, f64),
        heat: Option<&str>,
        pi_z: f64,
    ) -> PredictionOut {
        let ws_like_scale = self.ws_scale;
        let raw = &self.raw;
        let p = physical(kind, &raw[..n_base(kind)], self.temp_max_k);
        let bias = match heat {
            Some(h) => self
                .result
                .internal
                .heat_index
                .iter()
                .find(|(hh, _)| hh == h)
                .map(|(_, idx)| {
                    if *idx < 0 {
                        0.0
                    } else {
                        let gi = n_base(kind) + *idx as usize;
                        raw[gi]
                    }
                })
                .unwrap_or(0.0),
            None => 0.0,
        };
        let mu = mean_log_life(kind, &p, temp_k, stress, &ws_like_scale) + bias;

        let nfree = self.raw.len() - n_base(kind) - 1;
        let mut jac = jacobian_at(kind, self, temp_k, stress);
        // 炉次若指定且自由：偏置参数对预测的导数 = 1
        if let Some(h) = heat {
            if let Some((_, idx)) = self
                .result
                .internal
                .heat_index
                .iter()
                .find(|(hh, _)| hh == h)
            {
                if *idx >= 0 {
                    jac[n_base(kind) + *idx as usize] = 1.0;
                }
            }
        }
        let var_params: f64 = jac
            .iter()
            .enumerate()
            .map(|(i, &ji)| {
                ji * (0..jac.len())
                    .map(|j| self.cov[i][j] * jac[j])
                    .sum::<f64>()
            })
            .sum();
        let s2 = self.result.sigma_dex.powi(2);
        let se_total = (var_params + s2).sqrt();
        let leverage = (var_params / s2).max(0.0); // 仅参数不确定性 / 噪声方差
        let _ = nfree;

        // 归一化包络距离（基于训练点的温度/应力极差）
        let tr = (env.temp_max_k - env.temp_min_k).max(1.0);
        let sr = (env.stress_max_mpa - env.stress_min_mpa).max(1.0);
        let dt = ((env.temp_min_k - temp_k).max(0.0) + (temp_k - env.temp_max_k).max(0.0)) / tr;
        let ds = ((env.stress_min_mpa - stress).max(0.0) + (stress - env.stress_max_mpa).max(0.0)) / sr;
        let dist = dt.hypot(ds);
        let inside = dist <= 1e-9;

        let outside_allowed = temp_k < allow_temp_k.0
            || temp_k > allow_temp_k.1
            || stress < allow_stress.0
            || stress > allow_stress.1;

        let mut boundary = Vec::new();
        if temp_k <= env.temp_min_k + 1e-9 {
            boundary.push("temp_min".into());
        }
        if temp_k >= env.temp_max_k - 1e-9 {
            boundary.push("temp_max".into());
        }
        if stress <= env.stress_min_mpa + 1e-9 {
            boundary.push("stress_min".into());
        }
        if stress >= env.stress_max_mpa - 1e-9 {
            boundary.push("stress_max".into());
        }

        let (risk, warning) = if outside_allowed {
            (
                "rejected".into(),
                Some("预测点超出用户允许外推范围，拒绝给出曲线延长值".into()),
            )
        } else if inside {
            ("interpolated".into(), None)
        } else {
            let level = if dist <= 0.1 {
                ("low", "紧邻实验包络边界，风险低但仍为外推")
            } else if dist <= 0.4 {
                ("medium", "已离开实验包络，模型形式风险上升")
            } else {
                ("high", "远离实验包络，预测仅为受限模型外推，不可用于验收")
            };
            (level.0.into(), Some(level.1.into()))
        };

        let mean_h = 10f64.powf(mu);
        PredictionOut {
            temp_k,
            stress_mpa: stress,
            mean_log10_h: mu,
            median_h: mean_h,
            pi_low_h: 10f64.powf(mu - pi_z * se_total),
            pi_high_h: 10f64.powf(mu + pi_z * se_total),
            pi_level: 1.0 - erfc(pi_z / std::f64::consts::SQRT_2),
            inside_envelope: inside,
            outside_allowed,
            risk,
            boundary_active: boundary,
            normalized_distance: dist,
            extrapolation_leverage: leverage,
            warning,
        }
    }
}

fn jacobian_at(kind: ModelKind, fo: &FitOutput, temp_k: f64, stress: f64) -> Vec<f64> {
    let nb = n_base(kind);
    let n = fo.raw.len() - 1;
    let eps = 1e-5;
    let mut j = vec![0.0; n];
    for i in 0..n {
        let mut xp = fo.raw.clone();
        let mut xm = fo.raw.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let pp = physical(kind, &xp[..nb], fo.temp_max_k);
        let pm = physical(kind, &xm[..nb], fo.temp_max_k);
        let mp = mean_log_life(kind, &pp, temp_k, stress, &fo.ws_scale);
        let mm = mean_log_life(kind, &pm, temp_k, stress, &fo.ws_scale);
        j[i] = (mp - mm) / (2.0 * eps);
    }
    j
}

/// 样本纳入规则：
/// - criterion 必须与拟合准则完全一致（不同失效准则绝不混组）；
/// - 恒载样本始终纳入（断裂 exact / 右删失 censored 分别处理）；
/// - 变载样本仅在 damage_model=true（选定累积损伤模型）时纳入。
pub fn select_observations(
    samples: &[crate::ingest::ValidSample],
    criterion: &str,
    damage_model: bool,
) -> (Vec<Obs>, Vec<(String, String)>) {
    let mut obs = Vec::new();
    let mut excluded = Vec::new();
    for s in samples {
        if s.criterion != criterion {
            excluded.push((s.id.clone(), format!("失效准则 {} != {}", s.criterion, criterion)));
            continue;
        }
        if s.load_kind == "variable" && !damage_model {
            excluded.push((s.id.clone(), "变载样本：未选定累积损伤模型，不参与拟合".into()));
            continue;
        }
        if s.load_kind == "variable" {
            let segs: Vec<Seg> = s
                .segments
                .iter()
                .map(|g| Seg {
                    temp_k: g.temp_c + 273.15,
                    stress: g.stress_mpa,
                    hours: g.hours_h,
                })
                .collect();
            obs.push(Obs {
                sample_id: s.id.clone(),
                heat: s.heat.clone(),
                kind: if s.status == "fractured" { ObsKind::Exact } else { ObsKind::Censored },
                temp_k: segs.last().unwrap().temp_k,
                stress: segs.last().unwrap().stress,
                y_time: s.time_h.log10(),
                segments: segs,
                variable: true,
            });
        } else {
            obs.push(Obs {
                sample_id: s.id.clone(),
                heat: s.heat.clone(),
                kind: if s.status == "fractured" { ObsKind::Exact } else { ObsKind::Censored },
                temp_k: s.temp_c.unwrap_or_else(|| s.segments.last().unwrap().temp_c) + 273.15,
                stress: s.stress_mpa.unwrap(),
                y_time: s.time_h.log10(),
                segments: vec![],
                variable: false,
            });
        }
    }
    (obs, excluded)
}
