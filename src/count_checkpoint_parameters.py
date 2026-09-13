import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count tensor elements in a PyTorch checkpoint state_dict without instantiating the model."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-yaml", default=None)
    return parser.parse_args()


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")


def count_state_dict_tensors(state_dict):
    import torch

    tensors = {
        key: value
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }
    floating_tensors = {
        key: value
        for key, value in tensors.items()
        if torch.is_floating_point(value) or torch.is_complex(value)
    }
    return {
        "State_Dict_Tensor_Elements": int(sum(value.numel() for value in tensors.values())),
        "State_Dict_Floating_Tensor_Elements": int(sum(value.numel() for value in floating_tensors.values())),
        "State_Dict_Tensor_Count": int(len(tensors)),
        "State_Dict_Floating_Tensor_Count": int(len(floating_tensors)),
    }


def main():
    args = parse_args()
    import torch
    import yaml

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    counts = {
        "checkpoint": str(args.checkpoint),
        **count_state_dict_tensors(state_dict),
    }
    print(yaml.safe_dump(counts, sort_keys=False).strip())
    if args.output_yaml:
        output_path = Path(args.output_yaml)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(counts, handle, sort_keys=False)


if __name__ == "__main__":
    main()
