use crate::models::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetIn {
    pub temp_value: f64,
    pub temp_unit: Option<String>,
    pub stress_mpa: f64,
    pub heat: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitRequest {
    pub model: String,
    pub criterion: String,
    #[serde(default)]
    pub damage_model: bool,
    #[serde(default = "default_shrink")]
    pub shrink: f64,
    #[serde(default)]
    pub min_temp_c: Option<f64>,
    #[serde(default)]
    pub max_temp_c: Option<f64>,
    #[serde(default)]
    pub min_stress_mpa: Option<f64>,
    #[serde(default)]
    pub max_stress_mpa: Option<f64>,
    #[serde(default)]
    pub targets: Vec<TargetIn>,
    #[serde(default)]
    pub name: Option<String>,
}

impl FitRequest {
    pub fn bounds_ok(&self) -> bool {
        for (lo, hi) in [
            (self.min_temp_c, self.max_temp_c),
            (self.min_stress_mpa, self.max_stress_mpa),
        ] {
            if let (Some(a), Some(b)) = (lo, hi) {
                if !a.is_finite() || !b.is_finite() || a >= b {
                    return false;
                }
            }
        }
        true
    }
}

fn default_shrink() -> f64 {
    1.0
}

fn sigmoid(u: f64) -> f64 {
    1.0 / (1.0 + (-u).exp())
}
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}
fn map_param(u: f64, lo: f64, up: f64) -> f64 {
    lo + (up - lo) * sigmoid(u.clamp(-35.0, 35.0))
}

/// erf 的 Abramowitz & Stegun 7.1.26 近似（|误差|<1.5e-7）
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn log_phi_cdf(z: f64) -> f64 {
    if z >= 6.0 {
        return 0.0;
    }
    if z > -6.0 {
        let x = 1.0 - erf(z / std::f64::consts::SQRT_2);
        (0.5_f64 * x).ln()
    } else {
        let z2 = z * z;
        -0.5 * z2 - (-z).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - 1.0 / z2
    }
}

fn log_phi_pdf(z: f64) -> f64 {
    -0.5 * (z * z + (2.0 * std::f64::consts::PI).ln())
}

trait MasterModel {
    fn n_master(&self) -> usize;
    fn master_names(&self) -> Vec<&'static str>;
    fn start(&self) -> Vec<f64>;
    fn bounds(&self) -> Vec<(f64, f64)>;
    fn mean(&self, m: &[f64], tk: f64, x: f64) -> f64;
    fn grad_mean(&self, m: &[f64], tk: f64, x: f64) -> Vec<f64>;
}

struct LmModel;
impl MasterModel for LmModel {
    fn n_master(&self) -> usize {
        4
    }
    fn master_names(&self) -> Vec<&'static str> {
        vec!["C_LM", "a0", "a1", "a2"]
    }
    fn start(&self) -> Vec<f64> {
        vec![
            logit(20.0 / 60.0),
            logit((19500.0 - 10000.0) / 20000.0),
            logit((-13.0 + 30.0) / 60.0),
            logit(0.01 / 0.05),
        ]
    }
    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 60.0), (10000.0, 30000.0), (-30.0, 30.0), (0.0, 0.05)]
    }
    fn mean(&self, m: &[f64], tk: f64, x: f64) -> f64 {
        (m[1] + m[2] * x + m[3] * x * x) / tk - m[0]
    }
    fn grad_mean(&self, _m: &[f64], tk: f64, x: f64) -> Vec<f64> {
        // 自变量 x 为原始应力 MPa（非 log），与数据生成口径一致
        vec![-1.0, 1.0 / tk, x / tk, x * x / tk]
    }
}

struct CxModel;
impl MasterModel for CxModel {
    fn n_master(&self) -> usize {
        4
    }
    fn master_names(&self) -> Vec<&'static str> {
        vec!["beta", "gamma", "q", "alpha"]
    }
    fn start(&self) -> Vec<f64> {
        vec![
            logit((-5.0 + 30.0) / 60.0),
            logit((-6.0 + 20.0) / 40.0),
            logit((0.02 + 0.5) / 1.0),
            logit((8000.0 + 20000.0) / 40000.0),
        ]
    }
    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-30.0, 30.0), (-20.0, 20.0), (-0.5, 0.5), (-20000.0, 20000.0)]
    }
    fn mean(&self, m: &[f64], tk: f64, x: f64) -> f64 {
        // 受限形式: log10 tf = beta + gamma*x + (q/2)*x² + alpha/T
        m[0] + m[1] * x + 0.5 * m[2] * x * x + m[3] / tk
    }
    fn grad_mean(&self, _m: &[f64], tk: f64, x: f64) -> Vec<f64> {
        vec![1.0, x, 0.5 * x * x, 1.0 / tk]
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct InclusionRow {
    pub specimen_id: String,
    pub heat: String,
    pub temp_c: f64,
    pub stress_mpa: Option<f64>,
    pub outcome: String,
    pub variable: bool,
    pub included: bool,
    pub reason: String,
}

struct Problem {
    model: Box<dyn MasterModel>,
    n_master: usize,
    n_heat: usize,
    shrink: f64,
    heats: Vec<String>,
    /// true: 自变量为 log10(应力)；false: 自变量为原始应力 MPa
    x_is_log: bool,
}

impl Problem {
    fn x_of(&self, stress_mpa: f64) -> f64 {
        if self.x_is_log {
            stress_mpa.log10()
        } else {
            stress_mpa
        }
    }
}

impl Problem {
    fn decode(&self, u: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
        let bnds = self.model.bounds();
        let mut master = Vec::with_capacity(self.n_master);
        for i in 0..self.n_master {
            let (lo, up) = bnds[i];
            master.push(map_param(u[i], lo, up));
        }
        let mut offs = vec![0.0];
        for i in 0..self.n_heat {
            offs.push(map_param(u[self.n_master + i], -2.0, 2.0));
        }
        let tau = u[self.n_master + self.n_heat].clamp(-8.0, 4.0).exp();
        (master, offs, tau)
    }

    fn heat_idx(&self, heat: &str) -> usize {
        // 排序后第一个炉次即基准（偏移固定为 0）
        self.heats
            .iter()
            .position(|h| h == heat)
            .unwrap_or(0)
    }

    fn penalty(&self, u: &[f64]) -> f64 {
        let mut p = 0.0;
        for i in 0..self.n_heat {
            let b = map_param(u[self.n_master + i], -2.0, 2.0);
            p += self.shrink * b * b;
        }
        p
    }
}

struct Eval {
    nll: f64,
    logl: f64,
    rows: Vec<ResidualRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidualRow {
    pub specimen_id: String,
    pub heat: String,
    pub heat_idx: usize,
    pub temp_c: f64,
    pub x: f64,
    pub variable: bool,
    pub censored: bool,
    pub observed: String,
    pub predicted_log10_h: f64,
    pub residual: Option<f64>,
    pub std_residual: Option<f64>,
    pub contribution: f64,
    pub censor_tail_probability: Option<f64>,
    pub cumulative_damage: Option<f64>,
}

fn evaluate(prob: &Problem, included: &[&Specimen], u: &[f64]) -> Eval {
    let (master, offs, tau) = prob.decode(u);
    let mut sum = 0.0;
    let mut rows = Vec::new();

    for s in included {
        let hi = prob.heat_idx(&s.heat);
        if !s.variable {
            let seg = &s.segments[0];
            let x = prob.x_of(seg.stress_mpa);
            let mu = prob.model.mean(&master, seg.temp_kelvin, x) + offs[hi];
            let y = s.time_h.log10();
            let z = (y - mu) / tau;
            let (nll, contrib, tail) = if s.outcome == "censored" {
                let tailp = log_phi_cdf(z).exp();
                (-log_phi_cdf(z), log_phi_cdf(z), Some(tailp.min(1.0 - 1e-12)))
            } else {
                (
                    -(log_phi_pdf(z) - tau.ln()),
                    log_phi_pdf(z) - tau.ln(),
                    None,
                )
            };
            sum += nll;
            rows.push(ResidualRow {
                specimen_id: s.specimen_id.clone(),
                heat: s.heat.clone(),
                heat_idx: hi,
                temp_c: s.temp_c,
                x,
                variable: false,
                censored: s.outcome == "censored",
                observed: format!("t={}h ({})", s.time_h, s.outcome),
                predicted_log10_h: mu,
                residual: if s.outcome == "censored" { None } else { Some(y - mu) },
                std_residual: if s.outcome == "censored" { None } else { Some(z) },
                contribution: contrib,
                censor_tail_probability: tail,
                cumulative_damage: None,
            });
        } else {
            // Robinson 线性累积损伤：每段寿命为共享 log10-σ 的对数正态，
            // D=Σ dur/10^mu；以一阶 delta 近似 D 的分布尺度。
            let mut d = 0.0;
            let mut dvar = 0.0;
            let mut mus = Vec::new();
            for seg in &s.segments {
                let x = prob.x_of(seg.stress_mpa);
                let mu = prob.model.mean(&master, seg.temp_kelvin, x) + offs[hi];
                let w = seg.duration_h * 10f64.powf(-mu);
                d += w;
                dvar += w * w * (tau * tau * std::f64::consts::LN_10.powi(2));
                mus.push((seg.temp_kelvin, x, mu, w));
            }
            let sd_d = dvar.max(1e-12).sqrt();
            let z = (d - 1.0) / sd_d;
            let (nll, contrib, tail) = if s.outcome == "censored" {
                (-log_phi_cdf(z), log_phi_cdf(z), Some(log_phi_cdf(z).exp()))
            } else {
                (
                    -(log_phi_pdf(z) - sd_d.ln()),
                    log_phi_pdf(z) - sd_d.ln(),
                    None,
                )
            };
            sum += nll;
            let (tk, x, mu, _) = mus[0];
            rows.push(ResidualRow {
                specimen_id: s.specimen_id.clone(),
                heat: s.heat.clone(),
                heat_idx: hi,
                temp_c: tk - 273.15,
                x,
                variable: true,
                censored: s.outcome == "censored",
                observed: if s.outcome == "censored" {
                    format!("D={:.3}<1，t={}h 后未断（删失）", d, s.time_h)
                } else {
                    format!("D=1，t={}h 断裂", s.time_h)
                },
                predicted_log10_h: mu,
                residual: None,
                std_residual: Some(z),
                contribution: contrib,
                censor_tail_probability: tail,
                cumulative_damage: Some(d),
            });
        }
    }

    let pen = prob.penalty(u);
    Eval {
        nll: sum + pen,
        logl: -sum,
        rows,
    }
}

fn neldermead<F: Fn(&[f64]) -> f64>(f: F, start: Vec<f64>, max_iter: usize) -> (Vec<f64>, f64) {
    let n = start.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(start.clone());
    for i in 0..n {
        let mut v = start.clone();
        v[i] += if start[i].abs() > 0.1 { 0.1 } else { 0.25 };
        simplex.push(v);
    }
    let mut fv: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    // 自适应参数（Gao & Han 2012）
    let alpha = 1.0;
    let gamma = 1.0 + 2.0 / n as f64;
    let rho = 0.75 - 1.0 / (2.0 * n as f64);
    let sigma = 1.0 - 1.0 / n as f64;

    for _ in 0..max_iter {
        let mut order: Vec<usize> = (0..n + 1).collect();
        order.sort_by(|&a, &b| fv[a].partial_cmp(&fv[b]).unwrap());
        let (lo, hi) = (order[0], order[n]);
        if fv[hi] - fv[lo] < 1e-10 {
            break;
        }
        let mut centroid = vec![0.0; n];
        for &idx in &order[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }
        let xr: Vec<f64> = centroid
            .iter()
            .zip(&simplex[hi])
            .map(|(c, h)| c + alpha * (c - h))
            .collect();
        let fr = f(&xr);
        if fr < fv[lo] {
            let xe: Vec<f64> = centroid
                .iter()
                .zip(&simplex[hi])
                .map(|(c, h)| {
                    let xr_j = c + alpha * (c - h);
                    c + gamma * (xr_j - c)
                })
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[hi] = xe;
                fv[hi] = fe;
            } else {
                simplex[hi] = xr;
                fv[hi] = fr;
            }
        } else if fr < fv[order[n - 1]] {
            simplex[hi] = xr;
            fv[hi] = fr;
        } else {
            if fr < fv[hi] {
                let xc: Vec<f64> = centroid
                    .iter()
                    .zip(&simplex[hi])
                    .map(|(c, h)| {
                        let xr_j = c + alpha * (c - h);
                        c + rho * (xr_j - c)
                    })
                    .collect();
                let fc = f(&xc);
                if fc < fr {
                    simplex[hi] = xc;
                    fv[hi] = fc;
                    continue;
                }
            } else {
                let xcc: Vec<f64> = centroid
                    .iter()
                    .zip(&simplex[hi])
                    .map(|(c, h)| c - rho * (c - h))
                    .collect();
                let fc = f(&xcc);
                if fc < fv[hi] {
                    simplex[hi] = xcc;
                    fv[hi] = fc;
                    continue;
                }
            }
            for &idx in &order[1..] {
                for j in 0..n {
                    simplex[idx][j] = simplex[lo][j] + sigma * (simplex[idx][j] - simplex[lo][j]);
                }
                fv[idx] = f(&simplex[idx]);
            }
        }
    }
    let mut best = 0;
    for i in 1..n + 1 {
        if fv[i] < fv[best] {
            best = i;
        }
    }
    (simplex[best].clone(), fv[best])
}

fn numeric_hessian(f: &dyn Fn(&[f64]) -> f64, x: &[f64], f0: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let h = 1e-4_f64;
    let mut hessian = vec![vec![0.0; n]; n];

    for i in 0..n {
        let mut xpp = x.to_vec();
        xpp[i] += h;
        let fpp = f(&xpp);
        let mut xmm = x.to_vec();
        xmm[i] -= h;
        let fmm = f(&xmm);
        hessian[i][i] = (fpp - 2.0 * f0 + fmm) / (h * h);
        for j in (i + 1)..n {
            let mut xa = x.to_vec();
            xa[i] += h;
            xa[j] += h;
            let mut xb = x.to_vec();
            xb[i] += h;
            xb[j] -= h;
            let mut xc = x.to_vec();
            xc[i] -= h;
            xc[j] += h;
            let mut xd = x.to_vec();
            xd[i] -= h;
            xd[j] -= h;
            let v = (f(&xa) - f(&xb) - f(&xc) + f(&xd)) / (4.0 * h * h);
            hessian[i][j] = v;
            hessian[j][i] = v;
        }
    }
    hessian
}

fn invert_pd(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-10 {
            return None;
        }
        a.swap(col, piv);
        inv.swap(col, piv);
        let p = a[col][col];
        for j in 0..n {
            a[col][j] /= p;
            inv[col][j] /= p;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            for j in 0..n {
                a[r][j] -= factor * a[col][j];
                inv[r][j] -= factor * inv[col][j];
            }
        }
    }
    Some(inv)
}

fn cov_estimate(f: &dyn Fn(&[f64]) -> f64, u: &[f64], f0: f64) -> (Vec<Vec<f64>>, bool) {
    let mut hess = numeric_hessian(f, u, f0);
    // 岭加载确保正定
    let mut ridge = 1e-6;
    for _ in 0..20 {
        if let Some(cov) = invert_pd(&hess) {
            let ok = (0..hess.len()).all(|i| cov[i][i] >= 0.0);
            if ok {
                return (cov, ridge <= 1e-6);
            }
        }
        for i in 0..hess.len() {
            hess[i][i] += ridge;
        }
        ridge *= 10.0;
    }
    (vec![vec![f64::NAN; hess.len()]; hess.len()], false)
}

#[derive(Debug, Clone, Serialize)]
pub struct Envelope {
    pub min_temp_c: f64,
    pub max_temp_c: f64,
    pub min_stress_mpa: f64,
    pub max_stress_mpa: f64,
    pub heats: Vec<String>,
    pub n_temps: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub input: serde_json::Value,
    pub heat: String,
    pub within_envelope: bool,
    pub distance_temp_c: f64,
    pub distance_stress_mpa: f64,
    pub normalized_distance: f64,
    pub allowed: bool,
    pub blocked_by: Option<String>,
    pub risk: String,
    pub boundary_active: Vec<String>,
    pub median_h: f64,
    pub pi_low_h: f64,
    pub pi_high_h: f64,
    pub leverage: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FitResult {
    pub model: String,
    pub criterion: String,
    pub damage_model: bool,
    pub params: Vec<ParamReport>,
    pub tau_log10h: f64,
    pub heat_offsets: Vec<ParamReport>,
    pub n_included: usize,
    pub n_ruptured: usize,
    pub n_censored: usize,
    pub n_variable: usize,
    pub inclusion: Vec<InclusionRow>,
    pub residuals: Vec<ResidualRow>,
    pub censoring: serde_json::Value,
    pub envelope: Envelope,
    pub predictions: Vec<Prediction>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub n_free_params: usize,
    pub shrink: f64,
    pub warnings: Vec<String>,
    pub leverage_cutoff: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParamReport {
    pub name: String,
    pub value: f64,
    pub se: Option<f64>,
    pub note: String,
}

fn select_specimens<'a>(
    all: &'a [Specimen],
    criterion: &str,
    damage_model: bool,
) -> (Vec<&'a Specimen>, Vec<InclusionRow>) {
    let mut inc = Vec::new();
    let mut rows = Vec::new();
    for s in all {
        let mut included = true;
        let mut reason = "纳入".to_string();
        if s.criterion != criterion {
            included = false;
            reason = format!("失效准则为 {}，与本次 {} 不同，不混合", s.criterion, criterion);
        } else if s.variable && !damage_model {
            included = false;
            reason = "变载样本但未启用累积损伤模型，按要求不参与拟合".to_string();
        }
        if included {
            inc.push(s);
        }
        rows.push(InclusionRow {
            specimen_id: s.specimen_id.clone(),
            heat: s.heat.clone(),
            temp_c: s.temp_c,
            stress_mpa: s.stress_mpa,
            outcome: s.outcome.clone(),
            variable: s.variable,
            included,
            reason,
        });
    }
    (inc, rows)
}

fn target_kelvin(t: &TargetIn, who: usize) -> Result<f64, String> {
    match &t.temp_unit {
        Some(u) if u.trim() == "C" => {
            if !t.temp_value.is_finite() || t.temp_value <= -273.15 {
                return Err(format!("目标 {who}: 摄氏度非法"));
            }
            Ok(t.temp_value + 273.15)
        }
        Some(u) if u.trim() == "K" => {
            if !t.temp_value.is_finite() || t.temp_value <= 0.0 {
                return Err(format!("目标 {who}: 开尔文非法"));
            }
            Ok(t.temp_value)
        }
        Some(u) => Err(format!("目标 {who}: 温度单位 {u:?} 不支持（仅 C/K）")),
        None => Err(format!("目标 {who}: 缺少温度单位，拒绝猜测")),
    }
}

pub fn fit(all: &[Specimen], req: &FitRequest) -> Result<FitResult, DomainError> {
    if !req.bounds_ok() {
        return Err(DomainError("应力外推边界不合法".into()));
    }
    let model: Box<dyn MasterModel> = match req.model.as_str() {
        "LM" => Box::new(LmModel),
        "CX" => Box::new(CxModel),
        other => {
            return Err(DomainError(format!(
                "未知模型 {other:?}，仅支持 LM（Larson-Miller）或 CX（自定义受限形式）"
            )))
        }
    };

    let (included, inclusion) = select_specimens(all, &req.criterion, req.damage_model);
    if included.len() < 4 {
        return Err(DomainError(format!(
            "准则 {} 下仅有 {} 个可纳入样本（至少需要 4 个）；请检查失效准则与损伤模型开关",
            req.criterion,
            included.len()
        )));
    }

    // 炉次：基准炉次取该组出现的第一个，其余带收缩偏移
    let mut heats: Vec<String> = Vec::new();
    for s in &included {
        if !heats.contains(&s.heat) {
            heats.push(s.heat.clone());
        }
    }
    heats.sort();
    let n_heat = heats.len() - 1;
    let n_master = model.n_master();

    let x_is_log = req.model == "CX";
    let mut warnings = Vec::new();
    let prob = Problem {
        model,
        n_master,
        n_heat,
        shrink: if req.shrink.is_finite() && req.shrink >= 0.0 {
            req.shrink
        } else {
            warnings.push("shrink 非法，回退为 1.0".into());
            1.0
        },
        heats: heats.clone(),
        x_is_log,
    };

    let refs: Vec<&Specimen> = included.iter().copied().collect();
    let obj = |u: &[f64]| evaluate(&prob, &refs, u).nll;

    let mut start = prob.model.start();
    start.extend(std::iter::repeat(0.0_f64).take(n_heat));
    start.push(0.0); // log tau

    let (u1, f1) = neldermead(&obj, start.clone(), 3000);
    let mut start2 = start.clone();
    for v in start2.iter_mut() {
        *v += 0.35;
    }
    start2[n_master + n_heat] = -0.5;
    let (u2, f2) = neldermead(&obj, start2, 3000);
    let (uhat, fmin) = if f2 < f1 { (u2, f2) } else { (u1, f1) };

    let eval = evaluate(&prob, &refs, &uhat);
    let (cov, exact_cov) = cov_estimate(&obj, &uhat, fmin);
    if !exact_cov {
        warnings.push("观测信息矩阵奇异，已使用岭加载估计参数协方差，区间偏保守".into());
    }

    let (master_hat, offs_hat, tau_hat) = prob.decode(&uhat);
    let bnds = prob.model.bounds();
    let mut params = Vec::new();
    for (i, name) in prob.model.master_names().iter().enumerate() {
        // 内部参数方差经 logistic 映射传播
        let se = if cov[0].iter().all(|v| v.is_finite()) {
            let p = (master_hat[i] - bnds[i].0) / (bnds[i].1 - bnds[i].0);
            let d = (bnds[i].1 - bnds[i].0) * p * (1.0 - p);
            Some((d * d * cov[i][i]).abs().sqrt())
        } else {
            None
        };
        let mut note = String::new();
        if uhat[i].abs() > 5.5 {
            note.push_str("参数贴近映射边界，可能数据不足；");
            warnings.push(format!("参数 {} 靠近边界，边界约束活跃", name));
        }
        params.push(ParamReport {
            name: (*name).to_string(),
            value: master_hat[i],
            se,
            note,
        });
    }

    let mut heat_offsets = Vec::new();
    for (i, h) in heats.iter().enumerate() {
        if i == 0 {
            heat_offsets.push(ParamReport {
                name: format!("heat_bias[{h}]"),
                value: 0.0,
                se: None,
                note: "基准炉次（固定为 0）".into(),
            });
        } else {
            let ui = n_master + i - 1;
            let se = if cov[0].iter().all(|v| v.is_finite()) {
                let p = (offs_hat[i] + 2.0) / 4.0;
                let d = 4.0 * p * (1.0 - p);
                Some((d * d * cov[ui][ui]).abs().sqrt())
            } else {
                None
            };
            heat_offsets.push(ParamReport {
                name: format!("heat_bias[{h}]"),
                value: offs_hat[i],
                se,
                note: format!("相对基准炉次 {} 的 log10(h) 偏移，岭收缩 λ={}", heats[0], prob.shrink),
            });
        }
    }

    let n_rup = refs.iter().filter(|s| s.outcome == "ruptured").count();
    let n_cen = refs.iter().filter(|s| s.outcome == "censored").count();
    let n_var = refs.iter().filter(|s| s.variable).count();
    let k = n_master + n_heat + 1;
    let aic = 2.0 * fmin + 2.0 * k as f64;

    // 实验包络：来自实际纳入拟合的样本
    let mut tmin = f64::INFINITY;
    let mut tmax = f64::NEG_INFINITY;
    let mut smin = f64::INFINITY;
    let mut smax = f64::NEG_INFINITY;
    let mut temps = std::collections::BTreeSet::new();
    for s in &refs {
        for seg in &s.segments {
            tmin = tmin.min(seg.temp_kelvin - 273.15);
            tmax = tmax.max(seg.temp_kelvin - 273.15);
            smin = smin.min(seg.stress_mpa);
            smax = smax.max(seg.stress_mpa);
        }
        temps.insert(((seg_temp(&s.segments[0])) * 10.0).round() as i64);
    }
    let envelope = Envelope {
        min_temp_c: tmin,
        max_temp_c: tmax,
        min_stress_mpa: smin,
        max_stress_mpa: smax,
        heats: heats.clone(),
        n_temps: temps.len(),
    };

    // 允许外推范围默认等于实验包络；用户可显式放宽
    let allow_tmin = req.min_temp_c.unwrap_or(tmin);
    let allow_tmax = req.max_temp_c.unwrap_or(tmax);
    let allow_smin = req.min_stress_mpa.unwrap_or(smin);
    let allow_smax = req.max_stress_mpa.unwrap_or(smax);
    let tspan = (tmax - tmin).max(1.0);
    let sspan = (smax - smin).max(1.0);

    // 杠杆度截断阈值
    let cutoff = 2.0 * k as f64 / refs.len() as f64;

    let mut predictions = Vec::new();
    for (ti, t) in req.targets.iter().enumerate() {
        let tk = match target_kelvin(t, ti + 1) {
            Ok(v) => v,
            Err(e) => return Err(DomainError(e)),
        };
        if !t.stress_mpa.is_finite() || t.stress_mpa <= 0.0 {
            return Err(DomainError(format!("目标 {}: 应力须为正", ti + 1)));
        }
        let tc = tk - 273.15;
        let x = prob.x_of(t.stress_mpa);
        let heat = t.heat.clone().unwrap_or_else(|| heats[0].clone());
        let hi = prob.heat_idx(&heat);
        let heat_known = heats.iter().any(|h| h == &heat);

        let dt = if tc < tmin {
            tmin - tc
        } else if tc > tmax {
            tc - tmax
        } else {
            0.0
        };
        let ds = if t.stress_mpa < smin {
            smin - t.stress_mpa
        } else if t.stress_mpa > smax {
            t.stress_mpa - smax
        } else {
            0.0
        };
        let ndist = ((dt / tspan).powi(2) + (ds / sspan).powi(2)).sqrt();
        let within = dt == 0.0 && ds == 0.0;

        let mut boundary_active = Vec::new();
        if tc < allow_tmin {
            boundary_active.push(format!("温度下限 {}°C", allow_tmin));
        }
        if tc > allow_tmax {
            boundary_active.push(format!("温度上限 {}°C", allow_tmax));
        }
        if t.stress_mpa < allow_smin {
            boundary_active.push(format!("应力下限 {}MPa", allow_smin));
        }
        if t.stress_mpa > allow_smax {
            boundary_active.push(format!("应力上限 {}MPa", allow_smax));
        }
        if !heat_known {
            boundary_active.push(format!("炉次 {heat} 未在训练数据中（按热效应均值+方差处理）"));
        }
        let allowed = boundary_active
            .iter()
            .filter(|b| b.contains("下限") || b.contains("上限"))
            .count()
            == 0;

        // 对内部参数向量的数值梯度
        let grad_u = numeric_grad_u(&prob, tk, x, hi, &uhat);
        let var_param: f64 = quad_form(&cov, &grad_u);
        let var_heat = if heat_known {
            0.0
        } else {
            offs_hat[1..]
                .iter()
                .map(|b| b * b)
                .sum::<f64>()
                / (n_heat.max(1) as f64)
        };
        let sd_total = (tau_hat * tau_hat + var_param + var_heat).max(1e-9).sqrt();
        let mu = prob.model.mean(&master_hat, tk, x) + offs_hat[hi];
        let median_h = 10f64.powf(mu);
        let z95 = 1.959964;
        let pi_low_h = 10f64.powf(mu - z95 * sd_total);
        let pi_high_h = 10f64.powf(mu + z95 * sd_total);

        // 杠杆度：主参数设计向量在样本设计空间中的 hat 对角近似
        let leverage = leverage_for(&prob, &master_hat, &refs, tk, x);
        let mut risk = if within {
            "low".to_string()
        } else if ndist < 0.25 {
            "medium".to_string()
        } else {
            "high".to_string()
        };
        if leverage > cutoff {
            risk = "high".to_string();
            boundary_active.push(format!("杠杆度 {:.3} 超过截断阈值 {:.3}", leverage, cutoff));
        }
        if params.iter().any(|p| p.note.contains("边界")) {
            risk = "high".to_string();
        }

        let blocked_by = if allowed {
            None
        } else {
            Some(format!("超出允许外推范围: {}", boundary_active.join("；")))
        };

        predictions.push(Prediction {
            input: serde_json::json!({
                "temp_c": (tc * 100.0).round() / 100.0,
                "temp_kelvin": (tk * 100.0).round() / 100.0,
                "stress_mpa": t.stress_mpa,
            }),
            heat,
            within_envelope: within,
            distance_temp_c: (dt * 100.0).round() / 100.0,
            distance_stress_mpa: (ds * 100.0).round() / 100.0,
            normalized_distance: (ndist * 1000.0).round() / 1000.0,
            allowed,
            blocked_by,
            risk,
            boundary_active,
            median_h: (median_h * 100.0).round() / 100.0,
            pi_low_h: (pi_low_h * 100.0).round() / 100.0,
            pi_high_h: (pi_high_h * 100.0).round() / 100.0,
            leverage: (leverage * 10000.0).round() / 10000.0,
        });
    }

    let censor_contrib: Vec<serde_json::Value> = eval
        .rows
        .iter()
        .filter(|r| r.censored)
        .map(|r| {
            serde_json::json!({
                "specimen_id": r.specimen_id,
                "meaning": "右删失：仅知寿命 > 截止时间，绝不在截止时刻当作断裂",
                "log_likelihood": (r.contribution * 1e6).round() / 1e6,
                "survival_probability_at_stop": r.censor_tail_probability,
                "cumulative_damage": r.cumulative_damage,
            })
        })
        .collect();

    Ok(FitResult {
        model: req.model.clone(),
        criterion: req.criterion.clone(),
        damage_model: req.damage_model,
        params,
        tau_log10h: tau_hat,
        heat_offsets,
        n_included: refs.len(),
        n_ruptured: n_rup,
        n_censored: n_cen,
        n_variable: n_var,
        inclusion,
        residuals: eval.rows,
        censoring: serde_json::json!({
            "n_censored": n_cen,
            "entries": censor_contrib,
        }),
        envelope,
        predictions,
        log_likelihood: (eval.logl * 1e6).round() / 1e6,
        aic: (aic * 1e6).round() / 1e6,
        n_free_params: k,
        shrink: prob.shrink,
        warnings,
        leverage_cutoff: (cutoff * 10000.0).round() / 10000.0,
    })
}

fn seg_temp(s: &Segment) -> f64 {
    s.temp_kelvin - 273.15
}

fn numeric_grad_u(prob: &Problem, tk: f64, x: f64, hi: usize, uhat: &[f64]) -> Vec<f64> {
    let h = 1e-5;
    let mut g = vec![0.0; uhat.len()];
    let mean_u = |u: &[f64]| -> f64 {
        let (master, offs, _tau) = prob.decode(u);
        prob.model.mean(&master, tk, x) + offs[hi]
    };
    let f0 = mean_u(uhat);
    for i in 0..uhat.len() {
        let mut up = uhat.to_vec();
        up[i] += h;
        let mut um = uhat.to_vec();
        um[i] -= h;
        g[i] = (mean_u(&up) - mean_u(&um)) / (2.0 * h);
    }
    let _ = f0;
    g
}

fn quad_form(cov: &[Vec<f64>], g: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..g.len() {
        for j in 0..g.len() {
            if cov[i][j].is_finite() {
                s += g[i] * cov[i][j] * g[j];
            }
        }
    }
    s.max(0.0)
}

/// 非线性杠杆度：g'(X'X)^-1 g 的经典 hat 近似，仅用主参数设计向量
fn leverage_for(
    prob: &Problem,
    master: &[f64],
    refs: &[&Specimen],
    tk: f64,
    x: f64,
) -> f64 {
    let p = prob.n_master;
    let mut xtx = vec![vec![0.0; p]; p];
    let mut rows = Vec::new();
    for s in refs.iter() {
        for seg in &s.segments {
            let g = prob
                .model
                .grad_mean(master, seg.temp_kelvin, prob.x_of(seg.stress_mpa));
            rows.push(g.clone());
            for i in 0..p {
                for j in 0..p {
                    xtx[i][j] += g[i] * g[j];
                }
            }
        }
    }
    let inv = match invert_pd(&xtx) {
        Some(v) => v,
        None => return f64::NAN,
    };
    let g0 = prob.model.grad_mean(master, tk, x);
    let mut h = 0.0;
    for i in 0..p {
        for j in 0..p {
            h += g0[i] * inv[i][j] * g0[j];
        }
    }
    h.max(0.0)
}

