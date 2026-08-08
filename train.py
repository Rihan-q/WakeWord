from livekit.wakeword import (
    load_config,
    run_generate,
    run_augment,
    run_extraction,
    run_train,
    run_export,
    run_eval,
)

# Load the config from YAML
# `livekit-wakeword setup --config hey_leo.yaml`, so setup and training
config = load_config("Hey_Leo.yaml")

# Run the pipeline stage by stage
run_generate(config)     # TTS synthesis + adversarial negatives
run_augment(config)      # Add noise, reverb, pitch shifts
run_extraction(config)   # Extract mel spectrograms + speech embeddings → .npy
run_train(config)        # 3-phase adaptive training -> output/hey_leo/hey_leo.pt
onnx_path = run_export(config)  # Export to ONNX -> output/hey_leo/hey_leo.onnx

# Evaluate the exported model
results = run_eval(config, onnx_path)
print(f"AUT={results['aut']:.4f}  FPPH={results['fpph']:.2f}  Recall={results['recall']:.1%}")
