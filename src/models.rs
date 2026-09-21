use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct DomainError(pub String);

impl std::fmt::Display for DomainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for DomainError {}

impl From<rusqlite::Error> for DomainError {
    fn from(e: rusqlite::Error) -> Self {
        DomainError(format!("数据库错误: {e}"))
    }
}
impl From<serde_json::Error> for DomainError {
    fn from(e: serde_json::Error) -> Self {
        DomainError(format!("JSON 错误: {e}"))
    }
}

pub type DResult<T> = Result<T, DomainError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentIn {
    pub stress_mpa: f64,
    pub duration_h: f64,
    pub temp_value: Option<f64>,
    pub temp_unit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecimenIn {
    pub specimen_id: String,
    pub heat: String,
    pub temp_value: f64,
    /// 必须显式给出 "C" 或 "K"，缺失即拒收，不做猜测
    pub temp_unit: Option<String>,
    pub stress_mpa: Option<f64>,
    pub time_h: f64,
    /// "ruptured" 已断裂 | "censored" 右删失
    pub outcome: String,
    /// 失效准则，例如 rupture / strain_1pct；不同准则不得混拟
    pub criterion: String,
    #[serde(default)]
    pub variable: bool,
    #[serde(default)]
    pub segments: Vec<SegmentIn>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub stress_mpa: f64,
    pub duration_h: f64,
    pub temp_kelvin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Specimen {
    pub id: i64,
    pub specimen_id: String,
    pub heat: String,
    pub temp_kelvin: f64,
    pub temp_c: f64,
    pub stress_mpa: Option<f64>,
    pub time_h: f64,
    pub outcome: String,
    pub criterion: String,
    pub variable: bool,
    pub segments: Vec<Segment>,
    pub note: Option<String>,
}

fn to_kelvin(value: f64, unit: &str, who: &str) -> DResult<f64> {
    match unit.trim() {
        "C" => {
            if !value.is_finite() || value <= -273.15 {
                return Err(DomainError(format!("{who}: 摄氏度 {value} 不合法（须 > -273.15°C）")));
            }
            Ok(value + 273.15)
        }
        "K" => {
            if !value.is_finite() || value <= 0.0 {
                return Err(DomainError(format!("{who}: 开尔文 {value} 不合法（须 > 0 K）")));
            }
            Ok(value)
        }
        other => Err(DomainError(format!(
            "{who}: 温度单位必须显式为 C 或 K，收到 {other:?}，拒绝猜测单位"
        ))),
    }
}

impl SpecimenIn {
    pub fn validate(&self) -> DResult<Specimen> {
        let who = format!("试样 {}", self.specimen_id);
        if self.specimen_id.trim().is_empty() {
            return Err(DomainError("试样编号为空".into()));
        }
        if self.heat.trim().is_empty() {
            return Err(DomainError(format!("{who}: 炉次为空")));
        }
        if self.criterion.trim().is_empty() {
            return Err(DomainError(format!("{who}: 失效准则为空")));
        }
        let unit = match &self.temp_unit {
            Some(u) if u.trim() == "C" || u.trim() == "K" => u.trim().to_string(),
            Some(u) => {
                return Err(DomainError(format!(
                    "{who}: 温度单位仅支持 C/K，收到 {u:?}"
                )))
            }
            None => {
                return Err(DomainError(format!(
                    "{who}: 缺少温度单位，摄氏度与开尔文不得靠猜测换算"
                )))
            }
        };
        let temp_kelvin = to_kelvin(self.temp_value, &unit, &who)?;
        if !self.time_h.is_finite() || self.time_h <= 0.0 {
            return Err(DomainError(format!("{who}: 时间须为正数")));
        }
        if self.outcome != "ruptured" && self.outcome != "censored" {
            return Err(DomainError(format!(
                "{who}: outcome 仅支持 ruptured/censored"
            )));
        }

        let mut segs: Vec<Segment> = Vec::new();
        if self.variable {
            if self.segments.len() < 2 {
                return Err(DomainError(format!(
                    "{who}: 变载样本至少需要 2 个载荷段"
                )));
            }
            let mut total = 0.0;
            for (i, s) in self.segments.iter().enumerate() {
                if !s.stress_mpa.is_finite() || s.stress_mpa <= 0.0 {
                    return Err(DomainError(format!("{who}: 第 {} 段应力须为正", i + 1)));
                }
                if !s.duration_h.is_finite() || s.duration_h <= 0.0 {
                    return Err(DomainError(format!("{who}: 第 {} 段持续时间须为正", i + 1)));
                }
                let tk = match (&s.temp_value, &s.temp_unit) {
                    (None, None) => temp_kelvin,
                    (Some(v), u) => match u {
                        Some(u) => to_kelvin(*v, u, &format!("{who} 第{}段", i + 1))?,
                        None => {
                            return Err(DomainError(format!(
                                "{who} 第 {} 段: 有温度数值却缺单位，拒绝猜测",
                                i + 1
                            )))
                        }
                    },
                    (None, Some(_)) => {
                        return Err(DomainError(format!(
                            "{who} 第 {} 段: 有温度单位却缺数值",
                            i + 1
                        )))
                    }
                };
                total += s.duration_h;
                segs.push(Segment {
                    stress_mpa: s.stress_mpa,
                    duration_h: s.duration_h,
                    temp_kelvin: tk,
                });
            }
            if (total - self.time_h).abs() > 1e-6 * self.time_h.max(1.0) {
                return Err(DomainError(format!(
                    "{who}: 变载各段时长之和 {total} 与样本时间 {} 不一致",
                    self.time_h
                )));
            }
        } else {
            match self.stress_mpa {
                Some(s) if s.is_finite() && s > 0.0 => {}
                _ => {
                    return Err(DomainError(format!(
                        "{who}: 恒载样本需要正的 stress_mpa"
                    )))
                }
            }
            segs.push(Segment {
                stress_mpa: self.stress_mpa.unwrap(),
                duration_h: self.time_h,
                temp_kelvin,
            });
        }

        Ok(Specimen {
            id: 0,
            specimen_id: self.specimen_id.trim().to_string(),
            heat: self.heat.trim().to_string(),
            temp_kelvin,
            temp_c: temp_kelvin - 273.15,
            stress_mpa: if self.variable { None } else { self.stress_mpa },
            time_h: self.time_h,
            outcome: self.outcome.clone(),
            criterion: self.criterion.trim().to_string(),
            variable: self.variable,
            segments: segs,
            note: self.note.clone(),
        })
    }
}
