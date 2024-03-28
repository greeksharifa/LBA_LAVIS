import requests
import base64


def call_vision_api(args, prompt, image_paths):
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.openai_api_key}"
    }
    
    content = [{"type": "text", "text": prompt}] #"What’s in this image?"
    
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": args.vision_detail,
                }
            })
    
    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": content,
        }],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()


def call_chat_api(args, prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.openai_api_key}"
    }
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

