from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# === Load merged model ===
MODEL_PATH = "models/medical_llm_merged_final"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# === Set up text generation pipeline ===
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# === Interactive prompt loop ===
print("🧠 Merged LLM Inference Ready! Ask your medical/general prompt.")
print("Type 'exit' anytime to stop.\n")

while True:
    instruction = input("🔶 Instruction: ").strip()
    if instruction.lower() == "exit":
        break

    user_input = input("🔷 Input: ").strip()
    if user_input.lower() == "exit":
        break

    full_prompt = f"""### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"""

    output = pipe(
        full_prompt,
        max_new_tokens=216,  
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )[0]['generated_text']

    
    if "### Response:" in output:
        cleaned_output = output.split("### Response:")[-1].strip()
    else:
        cleaned_output = output.strip()

    print(f"\n🧾 Response:\n{cleaned_output}\n")
