"""Download one immutable model snapshot; safe alongside bootstrap via HF locks."""
from huggingface_hub import snapshot_download

snapshot_download(
    'autotrust/JEV-27B',
    revision='14b425269ed24452e0a5ad5a8f3e51d139647a88',
    local_dir='/root/autodl-tmp/JEV-27B', max_workers=4,
    allow_patterns=['model*.safetensors', 'model.safetensors.index.json',
                    'config.json', 'tokenizer*', 'chat_template.jinja',
                    'adapter_vllm/*', 'calibration.json', 'judge_config.json',
                    'serve_decide.py'],
)
print('PINNED_MODEL_DOWNLOAD_COMPLETE', flush=True)
