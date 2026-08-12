import torch
import os
from safetensors.torch import save_file

def convert_bin_to_safetensors_with_shared_tensors():
    """
    PyTorch의 .bin 모델 파일을 .safetensors 형식으로 변환합니다.
    이때, 메모리를 공유하는 텐서(tied weights)를 올바르게 처리합니다.
    """
    model_dir = "models/skt-kogpt2-base-v2"
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    safetensors_path = os.path.join(model_dir, "model.safetensors")

    if not os.path.exists(bin_path):
        print(f"❌ 모델 파일이 없습니다: {bin_path}")
        return

    print(f"'{bin_path}' 파일을 '{safetensors_path}'로 변환 중 (공유 텐서 처리)...")

    try:
        state_dict = torch.load(bin_path, map_location="cpu")
        
        # KoGPT-2는 wte와 lm_head 가중치를 공유합니다.
        # 이 정보를 메타데이터에 추가하여 safetensors가 올바르게 처리하도록 합니다.
        metadata = {'format': 'pt'}
        
        # save_file 함수는 state_dict의 텐서들이 공유하는 메모리가 있을 경우, 
        # 자동으로 처리하지 못하고 에러를 발생시킵니다.
        # 따라서 공유 관계를 명시적으로 알려주어야 합니다.
        # KoGPT2의 경우 lm_head.weight와 transformer.wte.weight가 동일한 스토리지를 공유합니다.
        # 이 관계를 명시하지 않고 저장하면, 로드 시 데이터가 깨질 수 있습니다.
        # 하지만, candle 라이브러리에서 로드할 때는 이 메타데이터를 직접 사용하지는 않으므로,
        # 저장 자체를 성공시키기 위해 빈 state_dict를 먼저 저장하고,
        # 실제 가중치는 clone()하여 저장하는 방식으로 우회할 수 있습니다.
        
        # 메모리 공유 문제를 해결하기 위해, 공유된 텐서를 복제합니다.
        if 'lm_head.weight' in state_dict and 'transformer.wte.weight' in state_dict:
             if state_dict['lm_head.weight'].data_ptr() == state_dict['transformer.wte.weight'].data_ptr():
                print("  - 공유 텐서 'lm_head.weight'와 'transformer.wte.weight'를 발견했습니다. lm_head.weight를 복제합니다.")
                state_dict['lm_head.weight'] = state_dict['lm_head.weight'].clone()

        save_file(state_dict, safetensors_path, metadata)
        
        print(f"✅ 변환 완료! 파일이 '{safetensors_path}'에 저장되었습니다.")

    except Exception as e:
        print(f"❌ 변환 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    convert_bin_to_safetensors_with_shared_tensors() 