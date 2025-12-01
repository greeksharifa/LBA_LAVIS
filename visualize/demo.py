import json
import gradio as gr
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import sys
import logging
import time
import random

# Add parent directory to path to import IndexSampler
sys.path.insert(0, str(Path(__file__).parent.parent))
from util.utils import IndexSampler

# Setup logger for IndexSampler (it uses logger internally)
logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs from IndexSampler


def load_and_filter_data(json_path: str) -> List[Dict]:
    """Load JSON data and filter: base_score == 0 and refined_score == 1"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    filtered_data = [
        item for item in data 
        if item.get('base_score') == 0 and item.get('refined_score') == 1
    ]
    
    return filtered_data


def extract_answer_letter(answer: str) -> str:
    """Extract answer letter from answer string (e.g., 'D. Kyungsu...' -> 'D')"""
    match = re.match(r'^([A-E])', answer.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return answer.strip().upper()


def simulate_refinement_process(sample: Dict, N: int = 5, M: int = 2, K: int = 4) -> Tuple[List[List[int]], List[float], str, int]:
    """
    Simulate the refinement process using IndexSampler:
    - Each refined answer uses M sub-QAs (default: 2)
    - Generate K refined answers (default: 4)
    - Continue until refined answer confidence >= 0.85 or K refined answers are generated
    - Return: (subq_pairs_list, confidences, final_answer, best_idx)
    
    Args:
        sample: Sample dictionary with subq_list, refined_answer_list, etc.
        N: Total number of sub-QAs (default: 5)
        M: Number of sub-QAs per refined answer (default: 2)
        K: Number of refined answers to generate (default: 4)
    """
    subq_list = sample.get('subq_list', [])
    confidences = sample['conf_refined']['token_min_prob']
    refined_answers = sample['refined_answer_list']
    base_conf = sample.get('conf_base', 0.0)
    threshold = 0.85
    
    # Use IndexSampler to get sub-QA pairs for each refined answer
    actual_N = len(subq_list) if subq_list else N
    actual_K = min(len(confidences), K)
    
    index_sampler = IndexSampler(actual_N, M, actual_K)
    subq_pairs_list = index_sampler.indices  # List of lists, e.g., [[0,1], [1,2], [2,3], [3,4]]
    
    # Track which refined answers were actually generated (until confidence threshold or K reached)
    generated_indices = []
    generated_confs = []
    
    for i in range(actual_K):
        generated_indices.append(i)
        generated_confs.append(confidences[i])
        
        # Check if confidence is high enough
        if confidences[i] >= threshold:
            break
    
    # Find the best answer (highest confidence among all refined answers)
    best_conf = max(confidences) if confidences else 0.0
    best_answer_idx = confidences.index(best_conf) if confidences else 0
    final_answer = refined_answers[best_answer_idx] if refined_answers else ""
    
    return subq_pairs_list, generated_confs, final_answer, best_answer_idx


def format_subq_pairs(subq_pairs_list: List[List[int]]) -> str:
    """Format sub-QA pairs as (1,2), (2,3), (3,4), (4,5)"""
    if not subq_pairs_list:
        return "None"
    
    pairs = []
    for pair in subq_pairs_list:
        # Convert 0-indexed to 1-indexed for display
        pair_1idx = [idx + 1 for idx in pair]
        pairs.append(f"({pair_1idx[0]}, {pair_1idx[1]})")
    
    return ", ".join(pairs) if pairs else "None"


def create_demo_interface(sample: Dict, idx: int) -> Tuple[str, str, str, str, str, List[str], List[float], str]:
    """Create the demo interface content for a sample"""
    
    # Extract information
    main_q = sample['main_q']
    vpath = sample['vpath']
    candidate_list = sample['candidate_list']
    base_answer = sample['base_answer']
    base_conf = sample.get('conf_base', 0.0)
    gt_ans = sample['gt_ans'].upper()
    
    # Simulate refinement process using IndexSampler
    subq_list = sample['subq_list']
    suba_list = sample['suba_list']
    N = len(subq_list)  # Total number of sub-QAs
    M = 2  # Number of sub-QAs per refined answer
    K = len(sample.get('refined_answer_list', []))  # Number of refined answers
    
    subq_pairs_list, confidences, final_refined_answer, best_idx = simulate_refinement_process(sample, N, M, K)
    subq_pairs = format_subq_pairs(subq_pairs_list)
    
    # Format base answer
    base_letter = extract_answer_letter(base_answer)
    base_correct = "❌ Wrong" if base_letter != gt_ans else "✅ Correct"
    
    # Format refined answer
    refined_letter = extract_answer_letter(final_refined_answer)
    refined_correct = "✅ Correct" if refined_letter == gt_ans else "❌ Wrong"
    max_conf = max(confidences) if confidences else 0.0
    
    # Determine which answer is better (refined should be better since base_score=0, refined_score=1)
    selected_answer = refined_letter
    selected_text = "Refined answer (better)"
    
    # Format candidate list
    candidate_text = "\n".join([f"{chr(65+i)}. {cand}" for i, cand in enumerate(candidate_list)])
    
    # Format ground truth answer
    # Find the candidate that matches the ground truth
    gt_letter = gt_ans.upper()
    gt_index = ord(gt_letter) - ord('A') if len(gt_letter) == 1 and gt_letter.isalpha() else -1
    if 0 <= gt_index < len(candidate_list):
        gt_answer_text = f"{gt_letter}. {candidate_list[gt_index]}"
    else:
        gt_answer_text = f"Ground Truth: {gt_ans}"
    
    # Format sub-QA process as list of steps for animation
    # Each step is structured with sub-components for sequential display
    subq_process_steps = []
    
    # Step 0: Base Answer (single step)
    step0 = []
    step0.append(f"Step 0: Base Answer")
    step0.append(f"  → Answer: {base_answer}")
    step0.append(f"  → Confidence: {base_conf:.4f} (Low confidence(<0.7), generating sub-QAs...)")
    step0.append("")
    subq_process_steps.append({
        'type': 'base',
        'parts': ["\n".join(step0)]
    })
    
    # Show each refined answer with its sub-QA pair
    # Each step is broken down into parts: header, sub-QA1 question, sub-QA1 answer, sub-QA2 question, sub-QA2 answer, refined answer, confidence
    for i, (subq_pair, conf) in enumerate(zip(subq_pairs_list[:len(confidences)], confidences)):
        # Convert 0-indexed to 1-indexed for display
        subq_pair_1idx = [idx + 1 for idx in subq_pair]
        
        # Build step parts for sequential display
        step_parts = []
        
        # Part 1: Step header
        header = f"Step {i+1}: Refined Answer {i+1} (using Sub-QA {subq_pair_1idx[0]}, {subq_pair_1idx[1]})\n"
        step_parts.append(header)
        
        # Part 2-5: Sub-QAs and their answers (one at a time)
        for j, subq_idx in enumerate(subq_pair):
            if subq_idx < len(subq_list):
                # Sub-QA question
                subq_question = f"  → Sub-QA {subq_pair_1idx[j]}: {subq_list[subq_idx]}\n"
                step_parts.append(subq_question)
                
                # Sub-QA answer
                if subq_idx < len(suba_list):
                    subq_answer = f"    Answer: {suba_list[subq_idx]}\n"
                    step_parts.append(subq_answer)
        
        # Part 6: Refined answer, confidence, and threshold (all together)
        refined_result = ""
        if i < len(sample['refined_answer_list']):
            refined_at_step = sample['refined_answer_list'][i]
            refined_result += f"  → Refined Answer: {refined_at_step}\n"
        
        refined_result += f"  → Confidence: {conf:.4f}\n"
        
        if conf >= 0.85:
            refined_result += f"  → ✓ Confidence threshold reached!\n"
        
        refined_result += "\n"
        step_parts.append(refined_result)
        
        subq_process_steps.append({
            'type': 'refined',
            'parts': step_parts,
            'confidence': conf
        })
    
    # Get sub-QAs used in best refined answer
    best_subq_pair = subq_pairs_list[best_idx] if best_idx < len(subq_pairs_list) else []
    best_subq_pair_1idx = [idx + 1 for idx in best_subq_pair]
    best_subq_text = f"Sub-QA {best_subq_pair_1idx[0]}, {best_subq_pair_1idx[1]}" if len(best_subq_pair_1idx) == 2 else "N/A"
    
    # Create comparison
    comparison = f"""
### Answer Comparison

**Base Answer:** {base_answer}  
Confidence: {base_conf:.4f} | Status: {base_correct}

**Best Refined Answer:** {final_refined_answer}  
Confidence: {max_conf:.4f} | Status: {refined_correct}  
Used Sub-QAs: {best_subq_text}

**Selected Answer:** **{selected_answer}** ({selected_text})
"""
    
    return (
        main_q,
        candidate_text,
        gt_answer_text,  # Ground truth answer
        base_answer,
        f"{base_conf:.4f}",
        subq_process_steps,  # Return as list of steps for animation
        confidences,  # Return confidences to check threshold
        comparison
    )


def animate_subq_process(subq_process_steps: List[str], delay: float = 1.0):
    """
    Generator function to animate sub-QA process step by step.
    Yields cumulative text with each step added.
    """
    cumulative_text = ""
    for step_text in subq_process_steps:
        cumulative_text += step_text
        yield cumulative_text
        time.sleep(delay)


def process_sample(data: List[Dict], sample_idx: int) -> Tuple:
    """Process a single sample and return all outputs"""
    if sample_idx < 0 or sample_idx >= len(data):
        return ("", "", "", "", "", "", "")
    
    sample = data[sample_idx]
    return create_demo_interface(sample, sample_idx)


def create_gradio_interface(json_path: str):
    """Create the Gradio interface"""
    
    # Load and filter data
    filtered_data = load_and_filter_data(json_path)
    
    if not filtered_data:
        print("No samples found matching the filter criteria (base_score=0, refined_score=1)")
        return None, []
    
    print(f"Loaded {len(filtered_data)} samples matching criteria")
    
    # Extract unique video directories for allowed_paths
    video_dirs = set()
    for sample in filtered_data:
        vpath = sample.get('vpath', '')
        if vpath:
            from pathlib import Path
            video_dir = str(Path(vpath).parent)
            video_dirs.add(video_dir)
    allowed_paths = list(video_dirs) if video_dirs else []
    
    # Create interface
    with gr.Blocks(title="Video QA Refinement Demo") as demo:
        gr.Markdown("# Video QA Refinement Demo")
        gr.Markdown("""
        This demo shows the refinement process for video question answering:
        1. Base answer is wrong, refined answer is correct
        2. Model generates sub-QAs when confidence is low (<0.7)
        3. Process continues until confidence >= 0.85 or 4 refined answers are generated
        4. Best refined answer is selected and compared with base answer
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                sample_idx = gr.Slider(
                    minimum=0,
                    maximum=len(filtered_data) - 1,
                    step=1,
                    value=0,
                    label="Sample Index",
                    info=f"Total samples: {len(filtered_data)}"
                )
                load_btn = gr.Button("Load Sample", variant="primary")
            
            with gr.Column(scale=2):
                video = gr.Video(label="Video", height=400)
        
        with gr.Row():
            with gr.Column():
                main_question = gr.Textbox(
                    label="Main Question",
                    lines=2,
                    interactive=False
                )
                candidates = gr.Textbox(
                    label="Candidate Answers",
                    lines=6,
                    interactive=False
                )
                gt_answer = gr.Textbox(
                    label="Ground Truth Answer",
                    lines=1,
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Answer (Initial Attempt)")
                base_answer = gr.Textbox(
                    label="Base Answer",
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Refinement Process (Sub-QAs)")
                subq_process = gr.Textbox(
                    label="Sub-QA Generation Process",
                    lines=15,
                    interactive=False
                )
            with gr.Column(scale=1):
                next_step_btn = gr.Button("Next Step", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Final Comparison")
                comparison = gr.Markdown()
        
        # State to track current step (starts at -1 so first Next Step shows Step 0)
        current_step = gr.State(-1)
        sample_data = gr.State(None)  # Store sample data
        
        # Define the update function
        def update_sample(idx, step_state):
            sample = filtered_data[int(idx)]
            outputs = create_demo_interface(sample, int(idx))
            
            # Get video path
            video_path = sample['vpath']
            
            # Reset step to -1 when loading new sample (so first Next Step shows Step 0)
            step_state = -1
            
            # Don't show any step initially - wait for Next Step button
            subq_process_steps = outputs[5]  # List of step dictionaries (updated index)
            confidences = outputs[6]  # List of confidences (updated index)
            current_text = ""  # Empty initially
            
            # Store sample data in state
            sample_state = {
                'subq_process_steps': subq_process_steps,
                'confidences': confidences,
                'comparison': outputs[7],  # Updated index
                'max_steps': len(subq_process_steps)
            }
            
            return (
                video_path,
                outputs[0],  # main_question
                outputs[1],  # candidates
                outputs[2],  # gt_answer
                outputs[3],  # base_answer
                current_text,  # subq_process (empty initially)
                "",  # comparison (empty initially)
                step_state,  # current_step
                sample_state  # sample_data (store the dict)
            )
        
        # Define next step function with sequential display
        def next_step(step_state, sample_state):
            # Handle None or invalid step_state
            if step_state is None:
                step_state = -1
            
            # Handle None or invalid sample_state
            if sample_state is None:
                yield "Please load a sample first.", "", step_state, sample_state
                return
            
            # Check if sample_state is a dict (it should be)
            if not isinstance(sample_state, dict):
                yield "Please load a sample first.", "", step_state, sample_state
                return
            
            subq_process_steps = sample_state.get('subq_process_steps', [])
            confidences = sample_state.get('confidences', [])
            comparison = sample_state.get('comparison', '')
            max_steps = sample_state.get('max_steps', 0)
            threshold = 0.85
            
            # Check if we have any steps
            if max_steps == 0 or len(subq_process_steps) == 0:
                yield "No steps available. Please load a sample first.", "", step_state, sample_state
                return
            
            # Move to next step (step_state starts at -1, so first click shows Step 0)
            next_step_idx = step_state + 1
            
            # Check if we've reached the end
            if next_step_idx >= max_steps:
                # Build full text from all steps
                full_text = ""
                for step in subq_process_steps:
                    if step['type'] == 'base':
                        full_text += "".join(step['parts'])
                    else:
                        full_text += "".join(step['parts'])
                yield full_text, comparison, next_step_idx, sample_state
                return
            
            # Get current step
            if next_step_idx < 0 or next_step_idx >= len(subq_process_steps):
                yield "Invalid step index.", "", step_state, sample_state
                return
            
            current_step = subq_process_steps[next_step_idx]
            
            # Build text up to current step
            full_text = ""
            for i in range(next_step_idx):
                step = subq_process_steps[i]
                if step['type'] == 'base':
                    full_text += "".join(step['parts'])
                else:
                    full_text += "".join(step['parts'])
            
            # Display each part sequentially with random delay
            cumulative_text = full_text
            for part in current_step['parts']:
                cumulative_text += part
                yield cumulative_text, "", next_step_idx, sample_state
                time.sleep(random.uniform(0.2, 0.5))
            
            # After all parts displayed, check if we should show comparison
            final_text = cumulative_text
            
            # Check if confidence threshold reached (for refined answer steps, index >= 1)
            if next_step_idx > 0 and current_step['type'] == 'refined':
                current_conf = current_step.get('confidence', 0.0)
                if current_conf >= threshold:
                    # Show final comparison
                    yield final_text, comparison, next_step_idx, sample_state
                    return
            
            # Check if we've reached max steps
            if next_step_idx >= max_steps - 1:
                yield final_text, comparison, next_step_idx, sample_state
                return
            
            yield final_text, "", next_step_idx, sample_state
        
        # Define reset function
        def reset_steps(sample_state):
            if sample_state is None or not isinstance(sample_state, dict):
                return "", "", -1, sample_state
            
            # Reset to -1 so Next Step will show Step 0
            return "", "", -1, sample_state
        
        load_btn.click(
            fn=update_sample,
            inputs=[sample_idx, current_step],
            outputs=[video, main_question, candidates, gt_answer, base_answer, subq_process, comparison, current_step, sample_data]
        )
        
        next_step_btn.click(
            fn=next_step,
            inputs=[current_step, sample_data],
            outputs=[subq_process, comparison, current_step, sample_data]
        )
        
        reset_btn.click(
            fn=reset_steps,
            inputs=[sample_data],
            outputs=[subq_process, comparison, current_step, sample_data]
        )
        
        # Auto-load first sample
        demo.load(
            fn=update_sample,
            inputs=[sample_idx, current_step],
            outputs=[video, main_question, candidates, gt_answer, base_answer, subq_process, comparison, current_step, sample_data]
        )
    
    return demo, allowed_paths


if __name__ == "__main__":
    json_path = "/home/ywjang/LBA_LAVIS/output/DramaQA/qwen3-vl-8b/refined_samples_list.json"
    
    demo, allowed_paths = create_gradio_interface(json_path)
    if demo:
        print(f"Allowed paths for Gradio: {allowed_paths}")
        # Allow Gradio to access video files in the data directory
        demo.launch(
            share=False, 
            server_name="0.0.0.0", 
            server_port=7860,
            allowed_paths=allowed_paths
        )
