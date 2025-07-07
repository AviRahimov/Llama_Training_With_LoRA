#!/usr/bin/env python3
"""
Clean Chat Test Script
Run this script to test the trained human-like model in a clean chat interface.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

def load_model():
    """Load the trained model and tokenizer"""
    print("🔄 Loading trained model...")
    
    # Configuration
    base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    adapter_path = "./llama3_rtx4060_lora_training/final_model_adapters_combined_datasets"
    
    # Check if adapter exists
    if not os.path.exists(adapter_path):
        print(f"❌ Error: Adapter path not found: {adapter_path}")
        print("Make sure you've completed the training and the adapter files exist.")
        return None, None
    
    # Load with 4-bit quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure model tokens
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
        base_model.generation_config.eos_token_id = tokenizer.eos_token_id
        
        # Load PEFT adapters
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def chat_with_human_model(model, tokenizer, conversation_history_str, max_new_tokens=150):
    """Generate a response from the human-like model"""
    
    # Ensure proper format
    if not conversation_history_str.strip().endswith("Human 2:"):
        if not conversation_history_str.strip() or conversation_history_str.strip().endswith("Human 1:"):
            pass
        elif conversation_history_str.strip().endswith("Human 1:"):
             conversation_history_str += "\nHuman 2: "
    
    # Add BOS token if needed
    if not conversation_history_str.startswith(tokenizer.bos_token):
        conversation_history_str = tokenizer.bos_token + conversation_history_str

    # Tokenize
    inputs = tokenizer(conversation_history_str, return_tensors="pt", padding=False, truncation=True).to(model.device)
    
    # Remove token_type_ids if present
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode response
    response_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # Clean up response
    response_text = response_text.strip()
    
    if response_text.startswith("Human 2:"):
        response_text = response_text[8:].strip()
    
    # Remove any partially generated next turns
    if "\nHuman 1:" in response_text:
        response_text = response_text.split("\nHuman 1:")[0]
    elif "Human 1:" in response_text:
        response_text = response_text.split("Human 1:")[0]
        
    return response_text.strip()

def clean_chat(model, tokenizer):
    """Clean chat interface with perfect ping-pong conversation flow"""
    print("\n🎭 HUMAN-LIKE AI CHAT")
    print("=" * 60)
    print("💬 You are chatting with a model trained to act like a human")
    print("🤖 The model was trained to forget it's an AI and respond naturally")
    print("📝 Type 'quit' or press Enter to end the conversation")
    print("\n" + "=" * 60 + "\n")
    
    conversation_history = []
    turn_number = 1
    
    while True:
        # Get user input
        user_message = input("You: ").strip()
        
        # Check for quit conditions
        if user_message.lower() in ['quit', 'exit', 'bye'] or user_message == "":
            print("\n👋 Chat ended. Thanks for the conversation!")
            break
        
        # Add user message to history
        conversation_history.append(f"Human 1: {user_message}")
        
        # Build conversation context
        conversation_context = "\n".join(conversation_history) + "\nHuman 2: "
        
        # Show thinking indicator
        print("Bot: 💭 thinking...", end="\r")
        
        try:
            # Get model response
            bot_response = chat_with_human_model(model, tokenizer, conversation_context, max_new_tokens=100)
            
            # Clean up response
            bot_response = bot_response.strip()
            
            # Ensure single response only
            if "\nHuman 1:" in bot_response:
                bot_response = bot_response.split("\nHuman 1:")[0].strip()
            if "\nHuman 2:" in bot_response:
                bot_response = bot_response.split("\nHuman 2:")[0].strip()
            
            # Display response
            print(f"Bot: {bot_response}" + " " * 20)
            
            # Add to history
            conversation_history.append(f"Human 2: {bot_response}")
            
            # Limit history length
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
            
            turn_number += 1
            
        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            print("Try asking something else...")
            continue
    
    print(f"\n📊 Conversation Summary:")
    print(f"   Total turns: {turn_number - 1}")
    print(f"   Messages exchanged: {len(conversation_history)}")
    print("\n✨ Hope you enjoyed chatting with your human-like AI!")

def main():
    """Main function"""
    print("🚀 Human-like AI Chat Interface")
    print("Loading model... This may take a moment...")
    
    # Load model
    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Start chat
    clean_chat(model, tokenizer)

if __name__ == "__main__":
    main()
