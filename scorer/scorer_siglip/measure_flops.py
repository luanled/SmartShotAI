import torch
from ptflops import get_model_complexity_info
from transformers import SiglipVisionModel

def measure_siglip_backbone():
    model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to("cpu")
    model.eval()

    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            (3, 224, 224),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )

    print("MACs:", macs)
    print("Params:", params)

if __name__ == "__main__":
    measure_siglip_backbone()
