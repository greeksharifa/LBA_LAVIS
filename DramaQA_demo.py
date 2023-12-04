import gradio as gr
import os
import json
import time

video_root_path = "/data1/AnotherMissOh/AnotherMissOh_videos/13~15"
qa_path = "/data1/AnotherMissOh/AnotherMissOhQA_val_set.json"

qa_data = json.load(open(qa_path, "r"))
qa_data = qa_data[4]
vid = qa_data["vid"]
print("vid:", vid) # AnotherMissOh14_001_0000


#
# # def greet(name):
# #     return "Hello " + name + "!!"
# # iface = gr.Interface(fn=greet, inputs="text", outputs="text")
# # iface.launch(share=True, server_name="0.0.0.0")
#
# def video_identity(video):
#     return video
#
#
# # with gr.Blocks() as demo:
# #     video_input = gr.Video(label="playable_video",
# #                            examples=[os.path.join(video_root_path, f"{vid}.mp4")],
# #                            cache_examples=True)
# #
# # iface = gr.Interface(video_identity,
# #                     gr.Video(),
# #                     "playable_video",
# #                     examples=[
# #                         os.path.join(video_root_path, f"{vid}.mp4")
# #                     ],
# #                     cache_examples=True)
#
# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column():
#             input_video = gr.Video(source=os.path.join(video_root_path, f"{vid}.mp4"), label="Upload Video")
#     with gr.Row():
#         main_question = gr.Textbox(label="Main Question")
#     with gr.Row():
#         with gr.Column():
#             sub_question_1 = gr.Textbox(label="Sub Question 1")
#         with gr.Column():
#             sub_answer_1 = gr.Textbox(label="Sub Answer 1")
#     with gr.Row():
#         with gr.Column():
#             sub_question_2 = gr.Textbox(label="Sub Question 2")
#         with gr.Column():
#             sub_answer_2 = gr.Textbox(label="Sub Answer 2")
#     with gr.Row():
#         main_answer = gr.Textbox(label="Main Answer")
#
#     process_btn = gr.Button("Process Video")
#
#     process_btn.click(
#         fn=video_identity,
#         inputs=input_video,
#         outputs=output_video
#     )
#

def get_sub_question(main_question, i):
    # main_Question: "How did Haeyoung1 feel when Haeyoung1 was with Dokyung?"
    time.sleep(0.5)
    sub_questions = [
        "What is the relationship between Haeyoung1 and Dokyung?",
        "What made Haeyoung1 smile while hugging the person?"
    ]
    return sub_questions[i]

def get_sub_answer(sub_question):
    time.sleep(0.3)
    sub_answers = {
        "What is the relationship between Haeyoung1 and Dokyung?": "Haeyoung1 and Dokyung are lovers.",
        "What made Haeyoung1 smile while hugging the person?": "Because Haeyoung1 was happy while hugging the person."
    }
    return sub_answers[sub_question]

def get_main_answer(main_question, sub_qas, num_sub_qa):
    time.sleep(0.5)
    main_answers = {
        0: "Haeyoung1 and Dokyung are happy.",
        1: "Haeyoung1 was happy because she loved Dokyung.",
        2: "Haeyoung1 was happy and laughed with Dokyung walking in the street."
    }
    return "Main Answer: " + main_answers[num_sub_qa]

def greet(video, main_question, num_sub_qa):
    num_sub_qa = int(num_sub_qa)
    sub_qas = ""
    for i in range(num_sub_qa):
        sub_question = get_sub_question(main_question, i)
        sub_answer = get_sub_answer(sub_question)
        sub_qas += f"sub_question {i+1:2d}: {sub_question}\n"
        sub_qas += f"sub_answer   {i+1:2d}: {sub_answer}\n"
    if num_sub_qa == 0:
        sub_qas = "No sub_qa created.\n"
    main_answer = get_main_answer(main_question, sub_qas, num_sub_qa)
    
    return sub_qas, main_answer

demo = gr.Interface(
    fn=greet,
    inputs=["video", "text", gr.Slider(0, 5, step=1)],
    outputs=["text", "text"],
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")

