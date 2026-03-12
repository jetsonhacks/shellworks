# Model Launch Scripts

This directory contains shell scripts for launching local model-serving configurations used by Shellworks.

Some launch scripts require a Hugging Face access token in order to download or access gated model repositories. In those cases, the `HF_TOKEN` environment variable must be set before running the script.

## Requirements

- Bash shell
- Required runtime dependencies installed for the target model
- Access to the model weights
- `HF_TOKEN` set when required by the target model

## Setting `HF_TOKEN`

Example:

```bash
export HF_TOKEN=your_huggingface_token
```

You may choose to export the token in your shell environment, source it from a local configuration file, or manage it through another secure workflow appropriate for your environment.

## Usage

Run a script directly, for example:

```bash
bash scripts/nemotron3-thor.sh
```

## Notes

- Not every script requires `HF_TOKEN`
- Some scripts may assume a specific local environment, model path, or container setup
- Review each script before use to understand its expected configuration and prerequisites
