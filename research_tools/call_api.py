import requests
import base64
import numpy as np
import cv2




def get_input(processor, device, video_batch, text_inputs):
    inputs = processor(text=text_inputs, return_tensors="pt", padding=True)

    pixel_values = []
    for video in video_batch: # video: [n_frms, 640, 480]
        pixel_values.append(processor(images=video, return_tensors="pt", padding=True)['pixel_values'])  # [n_frms, 3, 224, 224]
    inputs["pixel_values"] = torch.stack(pixel_values, dim=0)#.to(device)
    inputs = inputs.to(device)
    print("input_ids:", inputs["input_ids"].shape, inputs["input_ids"].device, "\tpixel_values:", inputs["pixel_values"].shape, inputs["pixel_values"].device)

    return inputs


def parse_args():
    parser = argparse.ArgumentParser(description='LBA method')
    parser.add_argument("--cfg-path", default='configs/runner.yaml', help="path to configuration file.")
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--chatgpt_mode', type=str, choices=["subq", "suba", "maina_before", "maina_after"])

    parser.add_argument('--chatgpt_model', type=str, default="gpt-4o-mini", help='OpenAI model')
    parser.add_argument('--chatgpt_max_tokens', type=int, default=500, help='OpenAI max tokens')
    parser.add_argument('--chatgpt_temperature', type=float, default=0., help='OpenAI temperature')
    parser.add_argument('--chatgpt_vision_detail', type=str, default="low", choices=["low", "high"])
    parser.add_argument('--chatgpt_verify', action='store_true', help='verify the generated questions')


    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args


def images_to_base64(image_list):
    """
    Convert a list of PIL.Image.Image objects to a list of base64-encoded strings.

    Args:
        image_list (list): List of PIL.Image.Image objects.

    Returns:
        list: List of base64-encoded strings.
    """
    base64_list = []
    for img in image_list:
        # Save the image to a BytesIO buffer
        buffer = BytesIO()
        img.save(buffer, format="PNG")  # Save as PNG (or another format, if preferred)
        buffer.seek(0)

        # Encode the buffer contents to base64
        base64_string = base64.b64encode(buffer.read()).decode("utf-8")
        base64_list.append(base64_string)
    return base64_list


def ndarrays_to_base64(frame_list):
    base64_frames = []

    for frame in frame_list:
        # Ensure the frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)

        # Convert the buffer to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        base64_frames.append(jpg_as_text)

    return base64_frames


def base64_to_ndarrays(base64_frames):
    decoded_frames = []

    for encoded in base64_frames:
        # Decode base64 string to bytes
        jpg_original = base64.b64decode(encoded)

        # Use numpy to create a 1D array of unsigned bytes
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)

        # Decode the JPEG to a numpy array
        frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)

        decoded_frames.append(frame)

    return decoded_frames


def get_subq_prompt(main_question):
    #     prompt = """Generate 5 perception questions based on given images, like following 2 examples:
    # Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?
    # Reasoning Question: is it cold outside? Perception Question: are any people wearing jackets?
    # Reasoning Question: {main_question}? Perception Question:"""

    prompt = """You will be provided with:
main_question: {main_question}
and video frames.

Your goal is:
To effectively analyze the video and answer the main_question, you should break down the main_question into several sub-questions that address the key aspects of the video.
You have to generate 5 sub_questions which those sub_questions can drive to the given main_question. sub_questions should be about the entire video.
Make sure that your sub_questions are based on the information you have.

Format: in list of dictionary format = ["sub_question_1", ..., "sub_question_5"]

sub_questions: """

    return prompt.format(main_question=main_question.rstrip('?'))

def get_suba_prompt(sub_questions):
    prompt = "Based on the video frames, answer the following questions. Important: Print answers only, one per one line.\n"
    for subq_i, sub_question in enumerate(sub_questions):
        prompt += f"Question {subq_i+1}: {sub_question}\n"
    prompt += "Answers: "

    return prompt


def get_maina_before_prompt(main_question, open_ended, candidate_list):
    # prompt = "Based on the video frames, answer the following question. Print the choice character only, such as '(A)'.\n"
    # prompt += f"Question: {main_question.rstrip('?')}?\n"

    # prompt += "Choices:\n"
    # for i, candidate in enumerate(candidate_list):
    #     prompt += f"({chr(65+i)}) {candidate}\n"

    # prompt += "Answer: The answer is "
    prompt = f"{main_question.rstrip('?')}?\n"
    if open_ended:
        prompt += "Answer the question using a single word or phrase."
    else:
        for i, candidate in enumerate(candidate_list):
            prompt += f"{chr(65+i)}. {candidate}\n"
        prompt += "Answer with the option's letter from the given choices directly."

    return prompt


def get_maina_after_prompt(main_question, sub_question_list, sub_answer_list, open_ended, candidate_list):
    # prompt = "Based on the video frames and additional sub-qa Context, answer the following question. Print the choice character only, such as '(A)'.\n"

    # prompt += "Context:\n"
    # for i, (sub_question, sub_answer) in enumerate(zip(sub_question_list, sub_answer_list)):
    #     prompt += f"sub-question {i+1}: {sub_question.rstrip('?')}? sub-answer {i+1}: {sub_answer.rstrip('.')}.\n"

    # prompt += f"Question: {main_question.rstrip('?')}?\n"

    # prompt += "Choices:\n"
    # for i, candidate in enumerate(candidate_list):
    #     prompt += f"({chr(65+i)}) {candidate}\n"

    # prompt += "Answer: The answer is "
    prompt = "Based on the video frames and additional sub-qa Context, answer the following question.\n"
    prompt += "Context:\n"
    for i, (sub_question, sub_answer) in enumerate(zip(sub_question_list, sub_answer_list)):
        prompt += f"sub-question {i+1}: {sub_question.rstrip('?')}? sub-answer {i+1}: {sub_answer.rstrip('.')}.\n"

    prompt += f"{main_question.rstrip('?')}?\n"
    if open_ended:
        prompt += "Answer the question using a single word or phrase."
    else:
        for i, candidate in enumerate(candidate_list):
            prompt += f"{chr(65+i)}. {candidate}\n"
        prompt += "Answer with the option's letter from the given choices directly."

    return prompt


reObj1 = re.compile(r'"(.+\?\'?)"')
reObj2 = re.compile(r'\d\. (.+\?\'?)')
def reformat_subq(text):
    result = reObj1.findall(text)
    if len(result) == 5 and all([r[-1] in "?'" for r in result]):
        return result

    result = reObj2.findall(text)
    if len(result) == 5 and all([r[-1] in "?'" for r in result]):
        return result

    return None

"""
Here are the sub-questions based on the provided video frames:

```json
[
    "What is the context of the conversation between the characters in the first two frames?",
    "How do the characters' expressions and body language indicate their feelings or reactions?",
    "What is the significance of the setting in the third frame, and how does it relate to the characters' interactions?",
    "What actions or dialogue occur just before the character enters the room in the fourth frame?",
    "How do the characters' clothing and colors contribute to the overall mood of the scenes?"
]

# -----------------------------------------------------------------------------------------------------------------------------------------------
Here are the sub-questions based on the provided video frames:

1. What is the context or setting of the event taking place in the video?
2. How are the characters interacting with each other in the frames?
3. What emotions or reactions are displayed by the characters in the video?
4. Are there any significant objects or elements in the environment that could influence the next event?
5. What prior events or conversations might have led to the current situation depicted in the video?

These sub-questions can help analyze the video and determine which event is more likely to happen next.
"""

def reformat_subq_backup(text):
    pattern = ''#r'.*'
    for i in range(1, 6):
        pattern += f'{i}\. \[?(.+\?)\]?.*'
        # pattern += f'{i}' + r'\. (?:\'\"\[)*(.+\?)(?:\'\"\])*.*'
        if i != 5:
            pattern += '\n'
    reObj = re.compile(pattern)
    import pdb; pdb.set_trace()

    # reObj = re.compile(r'1\. (.+\?).*\n2\. (.+\?)\n3\. (.+\?)\n4\. (.+\?)\n5\. (.+\?).*')
    matchObj = reObj.search(text)
    if matchObj:
        return [matchObj.group(i) for i in range(1, 6)]
    else:
        return None


reObj3 = re.compile(r'"(.+\'?)"')
reObj4 = re.compile(r'\d[\.:] (.+)')
def reformat_suba(text):
    result = reObj3.findall(text)
    if len(result) == 5:
        return [r.strip() for r in result]

    result = reObj4.findall(text)
    if len(result) == 5:
        return [r.strip() for r in result]

    return None




def call_openai_vision_api(args, prompt, base64_frames, system_prompt=""):
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.chatgpt_openai_api_key}"
    }

    content = []
    # content = [{
    #     "type": "text",
    #      "text": "Answer the following questions based on the video frames, respectively."
    # }] # ex. "What’s in this image?"

    for base64_frame in base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_frame}",
                "detail": args.chatgpt_vision_detail,
            }
        })

    content.append({
        "type": "text",
         "text": prompt
    })

    payload = {
        "model": args.chatgpt_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        "max_tokens": args.chatgpt_max_tokens,
        "temperature": args.chatgpt_temperature,
        "logprobs": True,
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()



def backup_call_vision_api(args, prompt, image_paths):
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.chatgpt_openai_api_key}"
    }

    content = [{"type": "text", "text": prompt}] # ex. "What’s in this image?"

    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": args.chatgpt_vision_detail,
                }
            })

    payload = {
        "model": args.chatgpt_model,
        "messages": [{
            "role": "user",
            "content": content,
        }],
        "max_tokens": args.chatgpt_max_tokens,
        "temperature": args.chatgpt_temperature,
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()


def call_openai_chat_api(args, prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.chatgpt_openai_api_key}"
    }
    payload = {
        "model": args.chatgpt_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.chatgpt_max_tokens,
        "temperature": args.chatgpt_temperature,
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()


def example():
    from openai import OpenAI
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    )

    print(completion.choices[0].message)
