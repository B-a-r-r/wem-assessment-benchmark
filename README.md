# wem-assessment-bechmark
This project aims to assess the artificial life model Ecology of Words as an open-endedness driven benchmark for LLMs.

Ecology of Words is a simulation model developed by Dr. Reiji SUZUKI and Dr. Takaya ARITA, both from the ALIFE-CORE laboratory at Nagoya University.


## Configuration
1. Jump into the project directory.
2. Run the setup scripts:
    - `setup.sh` for Linux
    - `setup.bat` for Windows

    <ins>NB</ins>: this scripts work for CUDA 11.8, please check your GPU compatbility and CUDA version to modify the requirements if needed.
3. Create a `.env` file in the root directory of the project and add a variable named `HF_AUTH_TOKEN` with your Hugging Face authentication token.

## About the LLMs used for the assessment
The LLMs used for the assessment are:
- `"deepseek-ai/Janus-Pro-1B"`
- `"mistralai/Mistral-7B-Instruct-v0.2"`
- `"google/gemma-3-4b-it"`
- `"Qwen/Qwen2.5-VL-3B-Instruct"`