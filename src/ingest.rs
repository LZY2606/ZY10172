//! 入库与数据口径校验。
//!
//! 铁律：绝对温度换算在入库时完成；摄氏度 (C) 与开尔文 (K) 必须显式声明，
//! 缺少单位或单位无法识别一律拒绝，绝不猜测。

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentIn {
    pub temp_value: f64,
    pub temp_unit: String,
    pub stress_mpa: f64,
    pub hours_h: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    pub id: String,
    pub heat: String,
    pub criterion: String,
    pub load_kind: String, // constant | variable
    #[serde(default)]
    pub temp_value: Option<f64>,
    #[serde(default)]
    pub temp_unit: Option<String>,
    #[serde(default)]
    pub stress_mpa: Option<f64>,
    pub time_h: f64,
    pub status: String, // fractured | censored
    #[serde(default)]
    pub segments: Vec<SegmentIn>,
    #[serde(default)]
    pub note: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredSegment {
    pub temp_c: f64,
    pub stress_mpa: f64,
    pub hours_h: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidSample {
    pub id: String,
    pub heat: String,
    pub criterion: String,
    pub load_kind: String,
    pub temp_c: Option<f64>,
    pub temp_k: Option<f64>,
    pub temp_unit: Option<String>,
    pub temp_value: Option<f64>,
    pub stress_mpa: Option<f64>,
    pub time_h: f64,
    pub status: String,
    pub segments: Vec<StoredSegment>,
    pub note: String,
}

/// 把 (数值, 单位) 转成摄氏度；缺单位/单位非法即报错。
pub fn to_celsius(value: f64, unit: Option<&str>, ctx: &str) -> Result<f64, String> {
    let unit = unit
        .map(|u| u.trim().to_ascii_uppercase())
        .ok_or_else(|| format!("{ctx}: 缺少温度单位，禁止在 C/K 之间猜测"))?;
    match unit.as_str() {
        "C" | "°C" | "CELSIUS" | "DEGC" => {
            if !(-273.15..=1200.0).contains(&value) {
                return Err(format!("{ctx}: 摄氏度 {value} 超出合理范围"));
            }
            Ok(value)
        }
        "K" | "KELVIN" => {
            if !(200.0..=2000.0).contains(&value) {
                return Err(format!(
                    "{ctx}: 开尔文 {value} 超出合理范围（疑似漏写摄氏度单位）"
                ));
            }
            Ok(value - 273.15)
        }
        other => Err(format!("{ctx}: 无法识别的温度单位 '{other}'（仅接受 C 或 K）")),
    }
}

pub fn validate(s: &Sample, known: &mut std::collections::HashSet<String>) -> Result<ValidSample, String> {
    if s.id.trim().is_empty() {
        return Err("样本 id 为空".into());
    }
    if !known.insert(s.id.clone()) {
        return Err(format!("样本 id 重复: {}", s.id));
    }
    if s.heat.trim().is_empty() {
        return Err(format!("{}: 炉次 heat 为空", s.id));
    }
    if s.criterion.trim().is_empty() {
        return Err(format!("{}: 失效准则 criterion 为空", s.id));
    }
    if !matches!(s.status.as_str(), "fractured" | "censored") {
        return Err(format!("{}: status 必须为 fractured 或 censored，实际为 {}", s.id, s.status));
    }
    if !(s.time_h.is_finite() && s.time_h > 0.0) {
        return Err(format!("{}: time_h 必须为正数（删失样本也只给截止时间）", s.id));
    }
    let mut seg_out = Vec::new();
    let mut tc = None;
    let mut tk = None;
    match s.load_kind.as_str() {
        "constant" => {
            let tv = s.temp_value.ok_or_else(|| format!("{}: 恒载样本缺少温度值", s.id))?;
            let c = to_celsius(tv, s.temp_unit.as_deref(), &s.id)?;
            let stress = s
                .stress_mpa
                .ok_or_else(|| format!("{}: 恒载样本缺少应力", s.id))?;
            if !stress.is_finite() || stress <= 0.0 {
                return Err(format!("{}: 应力必须为正", s.id));
            }
            tc = Some(c);
            tk = Some(c + 273.15);
        }
        "variable" => {
            if s.segments.is_empty() {
                return Err(format!("{}: 变载样本必须给出至少一个载荷段", s.id));
            }
            for (i, g) in s.segments.iter().enumerate() {
                let ctx = format!("{} 段{i}", s.id);
                let c = to_celsius(g.temp_value, Some(g.temp_unit.as_str()), &ctx)?;
                if !(g.stress_mpa.is_finite() && g.stress_mpa > 0.0) {
                    return Err(format!("{ctx}: 应力必须为正"));
                }
                if !(g.hours_h.is_finite() && g.hours_h > 0.0) {
                    return Err(format!("{ctx}: 段历时必须为正"));
                }
                seg_out.push(StoredSegment {
                    temp_c: c,
                    stress_mpa: g.stress_mpa,
                    hours_h: g.hours_h,
                });
            }
        }
        other => return Err(format!("{}: load_kind 非法 '{other}'", s.id)),
    }
    Ok(ValidSample {
        id: s.id.clone(),
        heat: s.heat.clone(),
        criterion: s.criterion.clone(),
        load_kind: s.load_kind.clone(),
        temp_c: tc,
        temp_k: tk,
        temp_unit: s.temp_unit.clone(),
        temp_value: s.temp_value,
        stress_mpa: s.stress_mpa,
        time_h: s.time_h,
        status: s.status.clone(),
        segments: seg_out,
        note: s.note.clone(),
    })
}
