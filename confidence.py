import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

def calculate_sentence_confidence(model, tokenizer, input_texts, generated_texts):
    # print("input_text: ", input_texts)
    # print("generated_text: ", generated_texts)
    
    # LBA TODO: 생성된 text에 대해서만 confidence를 계산하도록 수정
    
    if type(input_texts) == str:
        input_texts = list(input_texts)
    if type(generated_texts) == str:
        generated_texts = list(generated_texts)
    
    sentence_probs = []
    for input_text, generated_text in zip(input_texts, generated_texts):
        
        # Encode the input and generated text
        input_ids = tokenizer.encode(input_text, padding="longest", return_tensors='pt')
        generated_ids = tokenizer.encode(generated_text, padding="longest", return_tensors='pt')
        
        # Combine input and generated text for context
        combined_ids = torch.cat((input_ids, generated_ids), dim=1)
        
        # Generate attention mask
        attention_mask = torch.ones(combined_ids.shape, dtype=torch.long)
        
        # print(input_ids.device)
        # print(attention_mask.device)
        # print(combined_ids.device)
        # print(model.device)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        combined_ids = combined_ids.to(model.device)
        
        # Get the model's outputs
        with torch.no_grad():
            outputs = model(input_ids=combined_ids, attention_mask=attention_mask, labels=combined_ids)
        
        # Extract logits and calculate log probabilities
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get the log probabilities for the generated tokens
        generated_log_probs = []
        for i in range(len(input_ids[0]), combined_ids.size(1)):
            token_id = combined_ids[0, i].item()
            token_log_prob = log_probs[0, i, token_id].item()
            generated_log_probs.append(token_log_prob)
        
        # Sum the log probabilities to get the log likelihood of the sequence
        sentence_log_prob = torch.tensor(generated_log_probs).sum()
        
        # Convert log likelihood to probability
        sentence_prob = torch.exp(sentence_log_prob).item()
        sentence_probs.append(sentence_prob)
    
    return sentence_probs

def main():
    # Load the pre-trained Flan-T5 model and tokenizer
    model_name = 'google/flan-t5-base'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    
    # Input text and text to be generated
    input_text = "The cat sat on the"
    generated_text = " mat."
    
    # Calculate confidence for the entire generated sentence
    sentence_confidence = calculate_sentence_confidence(model, tokenizer, input_text, generated_text)
    
    # Print the sentence confidence
    print(f"Generated Sentence: {generated_text}")
    print(f"Sentence Confidence: {sentence_confidence:.4f}")

if __name__ == "__main__":
    main()
