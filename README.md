# RobotDesign1M Dataset

This repository provides:
- A subset of sample data from the **RobotDesign1M** dataset
- Code with prompts to generate instruction following data as stated in the paper
- A script and configuration to finetune the Qwen2VL model used in the paper

You can access the shared data here:  
👉 [Google Drive Folder](https://drive.google.com/drive/folders/1m1b1w6ROHci9xRWV-xSUA-Z8lXDXuDIk?usp=drive_link)

---

⚠️ **Note:** This repository is partially publicized and will be updated in the future. This repository only hosts a sample subset for demonstration purposes. The **RobotDesign1M** dataset will be released to support future research and advancements in AI-driven robotic design automation.

## Installation

1. Clone this repository:
```bash
cd robotdesign1m-public
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Text Generation

To run the text generation script that processes the raw retrieved data:

```bash
python generate_text.py
```

This script will:
- Load the model ([meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)).
- Process the `retrieved_texts.json` file. You can replace this file with your own data.
- Extract sentences and generate questions.
- Save the output to the `gen_output/` directory.

### Fine-tuning Qwen2VL

To fine-tune the Qwen2VL model on the RobotDesign1M dataset:

1. Clone the LLaMA-Factory submodule:
```bash
git submodule update --init --recursive
```

2. Update the configuration file `config/qwen2vl_full_sft_robotdesign1m.yaml` with your paths:
   - Set `model_name_or_path` to your pretrained model path
   - Set `output_dir` to your desired output directory

3. Run the fine-tuning script:
```bash
bash scripts-mllm-finetune/finetune_qwen2vl_robotdesign1m.sh
```

This will start the fine-tuning process using the specified configuration. Make sure you have sufficient GPU memory and the required CUDA devices available.

## Acknowledgement

We thank the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [vLLM](https://github.com/vllm-project/vllm) project for providing the frameworks used in this work. 
