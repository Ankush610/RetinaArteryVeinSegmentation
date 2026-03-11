import torch

ckpt = torch.load("./files/checkpoint.pth", map_location="cpu")

print("Epoch:", ckpt["epoch"])
print("Loss:", ckpt["loss"])

# Peek inside model weights
for k, v in ckpt["model_state_dict"].items():
    print(k, v.shape)
    break  # remove break to see all

# Peek optimizer state
print(ckpt["optimizer_state_dict"]["param_groups"])
