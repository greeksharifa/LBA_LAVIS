import json
import logging
import pickle
import time
from pprint import pprint

import gradio as gr

from sentence_transformers import SentenceTransformer, util
from torch import nn

video_root_path = "/data1/AnotherMissOh/AnotherMissOh_videos/13~15"
qa_path = "/data1/AnotherMissOh/AnotherMissOhQA_val_set.json"
KG_path = "/data1/AnotherMissOh/KnowledgeGraph_hanyang/KnowledgeGraph_threshold_200_trainval/val"
node_edge_path = "/data1/AnotherMissOh/KnowledgeGraph_hanyang/KG_Sample"

QID = 13032

qa_data = json.load(open(qa_path, "r"))
qa = None
for data in qa_data:
    if data["qid"] == QID:
        qa = data
        break
vid = qa["vid"]
# print("vid:", vid) # AnotherMissOh14_036_0000
print("qa:")
pprint(qa)

KG_data = pickle.load(open(f"{KG_path}/scene_{QID}.pkl", "rb"))
node_data = json.load(open(f"{node_edge_path}/node.json", "r"))
edge_data = json.load(open(f"{node_edge_path}/edge.json", "r"))
print(edge_data)
for kg in KG_data:
    concepts = kg["concepts"]
    for edge_index, edge_attribute in zip(kg["edge_index"], kg["edge_attribute"]):
        print(f"{node_data[str(concepts[edge_index[0].item()])]} \t | {edge_data[str(edge_attribute.item())]} \t | {node_data[str(concepts[edge_index[1].item()])]}")

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
"""
[{'concepts': array([  1380,   3877,   5926,  40703, 421335,   5926,   3418],
      dtype=int32), 'edge_attribute': tensor([15, 15, 15, 15]), 'edge_index': tensor([[2, 6],
        [3, 6],
        [4, 1],
        [5, 6]])},
{'concepts': array([1380, 3877, 5926, 1360], dtype=int32), 'edge_attribute': tensor([], dtype=torch.int64), 'edge_index': tensor([], size=(0, 2), dtype=torch.int64)},
{'concepts': array([  1380,   3877,   5926,   4495,   8420, 682725, 785418,   5354],
      dtype=int32), 'edge_attribute': tensor([15, 15, 15, 15]), 'edge_index': tensor([[0, 7],
        [2, 7],
        [3, 7],
        [4, 7]])},
{'concepts': array([  5926,    565,   1380,   3877,  13844,  14293,  15375,  34170,
       207211, 404519, 421335], dtype=int32), 'edge_attribute': tensor([ 5, 15, 15, 15, 15]), 'edge_index': tensor([[ 7,  4],
        [ 1,  4],
        [ 7,  4],
        [ 8,  6],
        [10,  3]])},
{'concepts': array([  1380,   3877,   5926,   1363,  11276,  11388,  21330,  22752,
        26755, 131898], dtype=int32), 'edge_attribute': tensor([ 1,  5, 15, 15, 15, 15]), 'edge_index': tensor([[8, 7],
        [7, 6],
        [1, 6],
        [4, 5],
        [5, 4],
        [6, 8]])}]
"""
"""
{'0': 'antonym', '1': 'atlocation', '2': 'capableof', '3': 'causes', '4': 'createdby', '5': 'isa', '6': 'desires', '7': 'hassubevent', '8': 'partof', '9': 'hascontext', '10': 'hasproperty', '11': 'madeof', '12': 'notcapableof', '13': 'notdesires', '14': 'receivesaction', '15': 'relatedto', '16': 'usedfor'}
hard     | relatedto     | close
hug      | relatedto     | close
pulled   | relatedto     | pull
hard     | relatedto     | close
arm      | relatedto     | like
hard     | relatedto     | like
dance    | relatedto     | like
street   | relatedto     | like
run_away         | isa   | run
away     | relatedto     | run
run_away         | relatedto     | run
tried    | relatedto     | try
pulled   | relatedto     | pull
police   | atlocation    | police_station
police_station   | isa   | station
pull     | relatedto     | station
need     | relatedto     | needed
needed   | relatedto     | need
station          | relatedto     | police
"""

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
    # main_Question: "Why did Dokyung pull Haeyoung1's arm hard?",
    time.sleep(0.5)
    sub_questions = [
        "Who is waving his hand with a smile?",
        "Who is about to hug Haeyoung1?",
        # "What is the relationship between Haeyoung1 and Dokyung?",
        # "What made Haeyoung1 smile while hugging the person?"
    ]
    return sub_questions[i]

def get_sub_answer(sub_question):
    time.sleep(0.2)
    sub_answers = {
        "Who is waving his hand with a smile?": "Haeyoung1 is waving her hand with a smile.",
        "Who is about to hug Haeyoung1?": "Dokyung is about to hug Haeyoung1.",
        # "What is the relationship between Haeyoung1 and Dokyung?": "Haeyoung1 and Dokyung are lovers.",
        # "What made Haeyoung1 smile while hugging the person?": "Because Haeyoung1 was happy while hugging the person."
    }
    return sub_answers[sub_question]

def get_main_answer(main_question, sub_qas, num_sub_qa):
    time.sleep(0.3)
    main_answers = {
        0: "Dokyung and Soontack are on the room.", #"Haeyoung1 and Dokyung are happy.",
        1: "Dokyung and Haeyoung1 were dancing on the street.", #"Haeyoung1 was happy because she loved Dokyung.",
        2: "Dokyung grabbed Haeyoung1's arm for a hug.", #"Haeyoung1 was happy and laughed with Dokyung walking in the street."
    }
    return "Main_Answer:\n" + main_answers[num_sub_qa]

def greet(Video, Main_Question, num_Sub_QA):
    num_Sub_QA = int(num_Sub_QA)
    
    # KG
    kg_data = KG_data[3]
    KG = "Knowledge_Graph:"
    concepts = kg_data["concepts"]
    for edge_index, edge_attribute in zip(kg_data["edge_index"], kg_data["edge_attribute"]):
        KG += f"\n({node_data[str(concepts[edge_index[0].item()])]} - "
        KG += f"{edge_data[str(edge_attribute.item())]} - "
        KG += f"{node_data[str(concepts[edge_index[1].item()])]})"
    if num_Sub_QA >= 1:
        KG += "\n({} - {} - {})".format("Haeyoung1", "wave", "hand")
    if num_Sub_QA >= 2:
        KG += "\n({} - {} - {})".format("Dokyung", "hug", "Haeyoung1")
       
    # Sub_QA
    sub_qas = ""
    for i in range(num_Sub_QA):
        sub_question = get_sub_question(Main_Question, i)
        sub_answer = get_sub_answer(sub_question)
        sub_qas += f"Sub_Q {i + 1:2d}: {sub_question}\n"
        sub_qas += f"Sub_A {i + 1:2d}: {sub_answer}\n"
    if num_Sub_QA == 0:
        sub_qas = "No Sub_QA created.\n"
        
    # Main_Answer
    main_answer = get_main_answer(Main_Question, sub_qas, num_Sub_QA)
    GT_answer = "Dokyung pulled Haeyoung1's arm to hug her hard."
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    embedding_main = sentence_transformer.encode(main_answer, convert_to_tensor=True)
    embedding_GT = sentence_transformer.encode(GT_answer, convert_to_tensor=True)
    similarity = cos_sim(embedding_main, embedding_GT)
    main_answer += f"\nSimilarity: {similarity:.3f}"

    return KG, sub_qas, main_answer

demo = gr.Interface(
    fn=greet,
    inputs=["video", "text", gr.Slider(0, 3, step=1)],
    outputs=["text", "text", "text"],
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")

