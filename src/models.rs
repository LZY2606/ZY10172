//! 受限参数形式的持久断裂主曲线。
//! 所有寿命在 log10 尺度建模：y = log10 t。
//!
//! * Larson-Miller：P = T_k*(C + y) = b0 + b1*s + b2*y
//!   => y = (T_k*C - b0 - b1*s)/(b2 - T_k)，C 固定 20，b2 受限 > Tmax。
//! * 自定义受限多项式：y = b0 + b1*s_norm + b2*s_norm^2 + b3*T_norm + b4*T_norm^2
//!   二次系数 b2,b4 被限定非正（防止包络外病态发散）。

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    LarsonMiller,
    Custom,
}

impl ModelKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "larson_miller" | "lm" => Some(Self::LarsonMiller),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }
}

/// 模型“物理参数”数量（不含炉次偏置、不含 log10_s）。
pub fn n_base(kind: ModelKind) -> usize {
    match kind {
        ModelKind::LarsonMiller => 3, // b0, b1, b2
        ModelKind::Custom => 5,       // b0..b4
    }
}

pub const LM_C: f64 = 20.0;

/// 把无约束优化变量映射为物理参数。
pub fn physical(kind: ModelKind, base: &[f64], temp_max_k: f64) -> Vec<f64> {
    match kind {
        ModelKind::LarsonMiller => vec![
            base[0],
            base[1],
            // b2 = temp_max + 50 + softplus(x2)，保证分母 b2 - T > 50
            temp_max_k + 50.0 + softplus(base[2]),
        ],
        ModelKind::Custom => vec![
            base[0],
            base[1],
            -softplus(base[2]), // b2 <= 0
            base[3],
            -softplus(base[4]), // b4 <= 0
        ],
    }
}

pub fn softplus(x: f64) -> f64 {
    if x > 40.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// 某一（温度, 应力）点的条件 log10 寿命均值。
pub fn mean_log_life(
    kind: ModelKind,
    p: &[f64],
    temp_k: f64,
    stress: f64,
    scale: &Scale,
) -> f64 {
    match kind {
        ModelKind::LarsonMiller => {
            (temp_k * LM_C - p[0] - p[1] * stress) / (p[2] - temp_k)
        }
        ModelKind::Custom => {
            let sn = (stress - scale.s0) / scale.sd;
            let tn = (temp_k - scale.t0) / scale.td;
            p[0] + p[1] * sn + p[2] * sn * sn + p[3] * tn + p[4] * tn * tn
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Scale {
    pub t0: f64,
    pub td: f64,
    pub s0: f64,
    pub sd: f64,
}

/// 参数显示元数据（供 UI/JSON 标注受限情况）。
pub fn param_meta(kind: ModelKind) -> Vec<(String, String, bool)> {
    match kind {
        ModelKind::LarsonMiller => vec![
            ("C".into(), "固定 Larson-Miller 常数".into(), false),
            ("b0".into(), "截距".into(), true),
            ("b1".into(), "应力一次项".into(), true),
            ("b2".into(), "受限: b2 > Tmax+50 K".into(), true),
        ],
        ModelKind::Custom => vec![
            ("b0".into(), "截距".into(), true),
            ("b1".into(), "应力一次项".into(), true),
            ("b2".into(), "受限: 应力二次项 <= 0".into(), true),
            ("b3".into(), "温度一次项".into(), true),
            ("b4".into(), "受限: 温度二次项 <= 0".into(), true),
        ],
    }
}

/// 参数初值（原始无约束尺度）。
pub fn initial_raw(kind: ModelKind) -> Vec<f64> {
    match kind {
        ModelKind::LarsonMiller => vec![0.0, 0.0, 0.0],
        ModelKind::Custom => vec![3.5, 0.0, -2.0, 0.0, -2.0],
    }
}
