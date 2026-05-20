# Qwen Fine-Tuning with Unsloth 🦥

This repository contains a complete, optimized pipeline (`qwen3_vl_finetune.py`) to fine-tune **Qwen** models (such as `unsloth/Qwen2-1.5B-Instruct-bnb-4bit` or `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit`) in 4-bit precision using **Unsloth** and **QLoRA**. 

The script is tailored for execution on **Google Colab** or **Kaggle** GPU runtimes and includes dataset processing, parameter-efficient fine-tuning, model saving/export, and structured JSON inference validation.

---

## 🚀 Key Features

* **Memory-Efficient Training**: Leverages Unsloth's 4-bit quantization and optimized Triton kernels to reduce memory footprint (up to 60% less VRAM) and speed up training (2× faster).
* **ShareGPT Format Support**: Easily maps dialogue datasets formatted in the standard ShareGPT schema into Hugging Face chat templates.
* **Structured JSON Extraction**: Implements prompt-seeding (`<|im_start|>assistant\n{`) to force the LLM to output clean, valid JSON responses for intent extraction without metadata headers or preamble chat.
* **Flexible Export Options**:
  * Save PEFT (LoRA) adapters locally.
  * Pack/zip adapters for direct browser download in Colab environment.
  * Options for 16-bit model merging and direct Hugging Face Hub pushes.

---

## 📋 Steps & Pipeline

1. **Environment Setup**: Uninstalls potentially conflicting packages and installs compatible CUDA 12.6 versions of PyTorch, torchvision, and Unsloth.
2. **Model Loading**: Loads pre-quantized 4-bit base models.
3. **LoRA Adapter Attachment**: Configures rank ($r=16$) and target modules (`q_proj`, `k_proj`, `v_proj`, etc.) with gradient checkpointing enabled.
4. **Dataset Parsing**: Reads a local `intent.json` dataset, converts roles (`human` $\rightarrow$ `user`, `gpt` $\rightarrow$ `assistant`), maps them into the tokenizer's chat template, and filters out empty dialogues.
5. **Supervised Fine-Tuning (SFT)**: Builds the `SFTTrainer` with memory optimizations (`adamw_8bit`, gradient accumulation, low per-device batch size).
6. **Adapter & Artifact Generation**: Saves model weights, zips the directory, and initiates local download.
7. **Post-Training Inference**: Tests intent extraction against a raw conversation transcript using custom formatting constraints.

---

## 💾 Dataset Format

Prepare your input dataset as `intent.json` in the same directory as the script. It must follow the **ShareGPT format**:

```json
[
  {
    "request_id": "20260220_81aa5822-4486-4e4c-b74b-2ffcbd64218e",
    "transcript": "Yes. What did you say? Good morning Ma'am, I had come to get my certificate for training. What is the name and what is the name of the college? Riyadh Riyadhwara Hoshiarpur I am not checking the certificates of Hoshiarpur. What are you doing? I had asked Muskan to call, did you come upstairs just now? Okay, that's your number. I'll give you their number, the second number. Send me a 'Hi' message on WhatsApp, I will send Muskan's number. Okay.",
    "timestamps": null,
    "diarized_transcript": {
      "entries": [
        {
          "transcript": "Yes.",
          "speaker_id": "0"
        },
        {
          "transcript": "What did you say?",
          "speaker_id": "1"
        },
        {
          "transcript": "Good morning",
          "speaker_id": "0"
        },
        {
          "transcript": "Ma'am, I had come to get my certificate for training.",
          "speaker_id": "1"
        },
        {
          "transcript": "What is the name and what is the name of the college?",
          "speaker_id": "0"
        },
        {
          "transcript": "Riyadh Riyadhwara Hoshiarpur",
          "speaker_id": "1"
        },
        {
          "transcript": "I am not checking the certificates of Hoshiarpur.",
          "speaker_id": "0"
        },
        {
          "transcript": "What are you doing?",
          "speaker_id": "1"
        },
        {
          "transcript": "I had asked Muskan to call, did you come upstairs just now?",
          "speaker_id": "0"
        },
        {
          "transcript": "Okay, that's your number. I'll give you their number, the second number.",
          "speaker_id": "1"
        },
        {
          "transcript": "Send me a 'Hi' message on WhatsApp, I will send Muskan's number.",
          "speaker_id": "0"
        },
        {
          "transcript": "Okay.",
          "speaker_id": "1"
        }
      ]
    },
    "language_code": "hi-IN",
    "language_probability": 0.956
  },
  {
    "request_id": "20260220_4265cb83-faf9-49fa-820c-469ec55c730b",
    "transcript": "Hello, yes friends. Hello, yes Prince. Hello, you said the dialogue, right? I am talking to you. Yes, sir. Hello And here too, they give a little less than two and a half to three lakh rupees. Yes, yes, yes. twenty twenty twenty-five four Hello, I am",
    "timestamps": null,
    "diarized_transcript": {
      "entries": [
        {
          "transcript": "Hello, yes friends.",
          "speaker_id": "0"
        },
        {
          "transcript": "Hello, yes Prince.",
          "speaker_id": "1"
        },
        {
          "transcript": "Hello, you said the dialogue, right?",
          "speaker_id": "0"
        },
        {
          "transcript": "I am talking to you.",
          "speaker_id": "0"
        },
        {
          "transcript": "Yes, sir.",
          "speaker_id": "1"
        },
        {
          "transcript": "Hello",
          "speaker_id": "0"
        },
        {
          "transcript": "And here too, they give a little less than two and a half to three lakh rupees.",
          "speaker_id": "1"
        },
        {
          "transcript": "Yes, yes, yes.",
          "speaker_id": "1"
        },
        {
          "transcript": "twenty twenty twenty-five four",
          "speaker_id": "1"
        },
        {
          "transcript": "Hello, I am",
          "speaker_id": "0"
        }
      ]
    },
    "language_code": "hi-IN",
    "language_probability": 0.987
  },
  {
    "request_id": "20260220_f0a109b2-09a7-4941-8161-75dd3d7f202c",
    "transcript": "Hello Hello Yes, Sachin. Ma'am, you had called to inquire about a flat or a PG. It will be with Bachittar Singh's friends. What, ma'am? Bachittar Singh told. You What? What? We were supposed to take two, but we just took them. I am saying, but Chitra told you.",
    "timestamps": null,
    "diarized_transcript": {
      "entries": [
        {
          "transcript": "Hello",
          "speaker_id": "0"
        },
        {
          "transcript": "Hello",
          "speaker_id": "1"
        },
        {
          "transcript": "Yes, Sachin.",
          "speaker_id": "0"
        },
        {
          "transcript": "Ma'am, you had called to inquire about a flat or a PG.",
          "speaker_id": "1"
        },
        {
          "transcript": "It will be with Bachittar Singh's friends.",
          "speaker_id": "0"
        },
        {
          "transcript": "What, ma'am?",
          "speaker_id": "1"
        },
        {
          "transcript": "Bachittar Singh told.",
          "speaker_id": "0"
        },
        {
          "transcript": "You",
          "speaker_id": "0"
        },
        {
          "transcript": "What?",
          "speaker_id": "1"
        },
        {
          "transcript": "What?",
          "speaker_id": "0"
        },
        {
          "transcript": "We were supposed to take two, but we just took them.",
          "speaker_id": "1"
        },
        {
          "transcript": "I am saying, but Chitra told you.",
          "speaker_id": "0"
        }
      ]
    },
    "language_code": "hi-IN",
    "language_probability": 0.979
  },
  {
    "request_id": "20260220_814855e8-c623-47b4-8ca3-c307367a5817",
    "transcript": "Hello Yes Ramandeep, how are you? Good afternoon. Good afternoon Yes, tell me. Yes, great. I am also fine, recognized? Yes, yes, I recognize you. Yes Okay, fine. When are your exams? Ma'am, it is until 24. It's until the 24th, man. I wanted to arrange your visit. How are all the students doing then? Yes Like my placement batch, its enrollment is going on. And as I had a conversation with Surender sir about your college. We have compensated the entire college in terms of fees. I had kept your fee at 12,000 rupees, right? For everyone. So in that, it's like, the seats and other things, the remaining children who are around it, like C.G.C., C.E., all the children are registering. So I wanted you all, once, whatever friends you are, to come and visit once and get your registration done on the same day. My which new We Yes, of everyone. So, I feel this. That's why I created a group for you, you might be in it. Yes, yes, yes, ma'am, it is. Yes, ma'am, I am here. So, I had a conversation with Charanpreet as well, but now he calls me Channpreet, Channpreet. Then, he hasn't even called me back today. Channapreet Then he/she to me Yes ma'am, our exams are going on. The exam that we had yesterday was a bit too high level, because of that we are all in a tense situation right now. Okay, so tomorrow you have that high-level paper again. No no, yesterday it aying for it, I am telling you about this course. Yes, okay, whatever changes, and the other two friends. Oh ma'am, I had to tell them, otherwise they were saying that we should check once after reaching home. They haven't even asked yet. Ma'am, maybe you spoke to them. They are from our college, Riya and Nikita. No, I haven't spoken to them, but Pratikya, from tomorrow I will be able to pay them only 4500, please sorry. Because you know I had given you the time for three or four days, right? Now we also have to respond to the company, you do it, I am doing your four. If theirs is done today, then it's four, tell them that from tomorrow they will come to 4500. Because you know. Yes Ma'am, those who have sent their scanner, their... Yes, do it on that, send one thousand and a screenshot. Okay. Okay, just update them once. Just update it along with the copy, okay? Yes, just update them that you will come in the same batch, ma'am is saying, so don't take so much time now and make the payment by this evening. Okay. Okay, Vidya.",
    "timestamps": null,
    "diarized_transcript": {
      "entries": [
      ]
    },
    "language_code": "hi-IN",
    "language_probability": 0.893
  }
]
```

---

## ⚙️ Requirements & Dependencies

The script automatically manages dependencies for **CUDA 12.6** environments (Colab/Kaggle). If running manually, the key packages installed are:

```bash
pip install unsloth_zoo
pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --upgrade bitsandbytes accelerate
pip install --upgrade torchvision==0.25.0+cu126 torchaudio==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

---

## 🛠️ Usage Instructions

### 1. Upload to Colab or Kaggle
Upload `qwen3_vl_finetune.py` along with your dataset `intent.json` to your workspace.

### 2. Configure Model and Paths
Open the script and adjust the model parameters if needed:
```python
# Select target base model
MODEL_NAME  = "unsloth/Qwen2-1.5B-Instruct-bnb-4bit" # Or "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048                                   # Sequence length limit
DATA_PATH   = "intent.json"                          # Path to ShareGPT dataset
```

### 3. Run the Script
Execute the script:
```bash
python qwen3_vl_finetune.py
```
> ⚠️ **Note on Notebooks/Colab:** If running inside an interactive notebook, run the installation step, **restart the runtime session** (Runtime $\rightarrow$ Restart session), and then run the remaining cells to prevent DLL or PyTorch device errors.

---

## 🎯 JSON Inference Seed Trick

Standard instruction models sometimes struggle to return *only* JSON, occasionally wrapping it in markdown blocks (e.g., ` ```json ... ``` `) or adding conversational preambles. 

This script implements a **seed trick** during evaluation to guarantee clean JSON:
```python
prompt = (
    f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    f"<|im_start|>user\n{TEST_TRANSCRIPT}<|im_end|>\n"
    f"<|im_start|>assistant\n{{"       # <-- Seed the response with the starting '{'
)
```
By forcing the model's generation to begin precisely after the opening bracket `{`, the model is steered to continue generating valid JSON key-value pairs directly. The post-processing logic then cleanly slices out the final brace:
```python
first_brace = raw.find("{")
last_brace  = raw.rfind("}")
response    = raw[first_brace : last_brace + 1]
```

---

## 💡 Hyperparameter Tuning & Tips

* **VRAM issues (OOM)**: 
  * Lower `per_device_train_batch_size` to `1`.
  * Increase `gradient_accumulation_steps` (e.g., `16` or `32`) to keep your effective batch size stable.
  * Reduce `MAX_SEQ_LEN` to `2048` or `1024`.
* **Exporting to Production**:
  * To export a full merged 16-bit model instead of just the LoRA adapter, uncomment Step 8b:
    ```python
    model.save_pretrained_merged("intent_extractor_merged_16bit", tokenizer, save_method="merged_16bit")
    ```
  * To upload weights to the Hugging Face Hub, set your `HF_TOKEN` environment variable and uncomment Step 8c.
