# LBA_SubQA

## Archive

Ignored artifacts recovered from the retired standalone checkout are preserved in [`archive/legacy_2024/`](archive/legacy_2024/). They include the March 2024 prototype, embedded API response examples, and its historical run command; no API credential is stored in the archive.



## Generate SubQA

- (24/03/28)
    - 현재 api는 ChatGPT를 사용
    - 모델은 `gpt-4-turbo-preview`가 기본값, vision 사용 시 `gpt-4-vision-preview`

- speech 추가 예정


### Usage

- 첫 사용 시 `generate_subqa/` 위치에 `api_key.json` 파일을 다음과 같은 내용으로 생성할 것(`dummy_api_key.json` 참고)
    - `{"LBA": "sk-...f7"}`
- `--debug` 옵션 추가 시 debug 모드로 작동, 사전에 설정된 `response_data`를 사용하며 결과를 파일로 저장하지 않음

```bash
cd generate_subqa

# Description + Knowledge Graph
python generate.py --prompt_path="prompts/subqa_240314.txt"

# plus vision(3 frames per 1 mainQA)
python generate.py --prompt_path="prompts/subqa_240327_vision.txt" --vision --vision_detail=low --max_vision_num=3 --output_dir="your_output_dir_path"

# plus speech(3 frames per 1 mainQA)
python generate.py --prompt_path="prompts/subqa_240328_vision_speech.txt" --vision --vision_detail=low --max_vision_num=3 --output_dir=/data1/AnotherMissOh/api_output/ --speech
```
