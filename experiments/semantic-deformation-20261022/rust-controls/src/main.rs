use std::io::{self, Read};

fn main() {
    let result = (|| -> Result<(), String> {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|e| e.to_string())?;
        let request = serde_json::from_str(&input).map_err(|e| format!("invalid request: {e}"))?;
        let output = semantic_deformation_controls::generate(&request)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?
        );
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("{}", serde_json::json!({"error": error}));
        std::process::exit(2);
    }
}
