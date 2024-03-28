# LBA_SubQA



## Generate SubQA

- (24/03/28)
    - 현재 api는 ChatGPT를 사용
    - gpt-4-turbo-preview가 기본값


- vision, speech 추가 예정


### Usage

- 첫 사용 시 `generate_subqa/` 위치에 `api_key.json` 파일을 다음과 같은 내용으로 생성할 것
    - `{"LBA": "sk-...f7"}`
- `--debug` 옵션 추가 시 debug 모드로 작동, 사전에 설정된 `response_data`를 사용하며 결과를 파일로 저장하지 않음

```bash
cd generate_subqa
python generate.py --prompt_path="prompts/subqa_240314.txt"
```