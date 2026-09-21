use mastercurve_bench::models::SpecimenIn;

fn base() -> serde_json::Value {
    serde_json::json!({
        "specimen_id":"U1","heat":"A","temp_value":550,"stress_mpa":100.0,
        "time_h":100.0,"outcome":"ruptured","criterion":"rupture","variable":false
    })
}

#[test]
fn temperature_units_are_validated_at_ingest() {
    // 缺单位：拒绝，不猜测
    let mut v = base();
    v.as_object_mut().unwrap().remove("temp_unit");
    let parsed: SpecimenIn = serde_json::from_value(v).unwrap();
    let err = parsed.validate().unwrap_err();
    assert!(err.to_string().contains("缺少温度单位"), "{}", err);

    // C 与 K 都显式给出时分别换算
    let mut vc = base();
    vc["temp_unit"] = "C".into();
    let s_c: SpecimenIn = serde_json::from_value(vc).unwrap();
    let specimen_c = s_c.validate().unwrap();
    assert!((specimen_c.temp_kelvin - 823.15).abs() < 1e-9);

    let mut vk = base();
    vk["temp_value"] = 823.15.into();
    vk["temp_unit"] = "K".into();
    let s_k: SpecimenIn = serde_json::from_value(vk).unwrap();
    let specimen_k = s_k.validate().unwrap();
    assert!((specimen_k.temp_c - 550.0).abs() < 1e-9);

    // 非法单位
    let mut vbad = base();
    vbad["temp_unit"] = "F".into();
    let sb: SpecimenIn = serde_json::from_value(vbad).unwrap();
    assert!(sb.validate().unwrap_err().to_string().contains("仅支持 C/K"));

    // 变载样本段时长之和必须等于样本时间
    let vv = serde_json::json!({
        "specimen_id":"V9","heat":"A","temp_value":550,"temp_unit":"C",
        "time_h":100.0,"outcome":"ruptured","criterion":"rupture","variable":true,
        "segments":[{"stress_mpa":120.0,"duration_h":40.0},
                    {"stress_mpa":90.0,"duration_h":50.0}]
    });
    let sv: SpecimenIn = serde_json::from_value(vv).unwrap();
    assert!(sv.validate().unwrap_err().to_string().contains("时长之和"));
}
