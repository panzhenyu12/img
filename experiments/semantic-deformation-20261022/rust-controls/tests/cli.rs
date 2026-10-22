use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn original_source_json_coordinates_survive_zero_edit_exactly() {
    let mut input: serde_json::Value =
        serde_json::from_str(include_str!("../examples/whole-width.json")).unwrap();
    input["amount"] = serde_json::json!(0);
    input["contour"][0]["source"] = serde_json::json!([450.5, 184.30899047851562]);
    let result = invoke(&input.to_string());
    assert!(result.status.success());
    let output: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        output["targets"][0]["target"][1].as_f64().unwrap(),
        184.30899047851562
    );
}

fn invoke(input: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_face-width-targets"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn cli_generates_json_and_rejects_unknown_motion_fields() {
    let input = include_str!("../examples/whole-width.json");
    let result = invoke(input);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        output["targets"][0]["target"],
        serde_json::json!([-70.4, 20.])
    );
    assert_eq!(
        output["part_rules"][0]["motion"]["translation"],
        serde_json::json!([-4.8, -0.])
    );
    let mut bad: serde_json::Value = serde_json::from_str(input).unwrap();
    bad["parts"][0]["motion"]["lock_position"] = serde_json::json!(true);
    let rejected = invoke(&bad.to_string());
    assert_eq!(rejected.status.code(), Some(2));
    assert!(rejected.stdout.is_empty());
    assert!(
        serde_json::from_slice::<serde_json::Value>(&rejected.stderr).unwrap()["error"].is_string()
    );
}
