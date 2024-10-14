
def map_prediction_to_answer(out):
    out = str(out).strip()
    out = out.replace('\u200b', '')
    out = out.replace('The answer is ', '')
    
    
    for i in range(2):
        if len(out) < i + 1:
            continue
        if out[i] in ["A", "B", "C", "D", "E"]:
            return '(' + out[i] + ')'
        if out[i] in list(range(5)):
            return '(' + chr(ord('A') + out[i]) + ')'
    
    for answer in ["A", "B", "C", "D", "E"]:
        if answer + ':' in out:
            return '(' + answer + ')'
        if answer + ')' in out:
            return '(' + answer + ')'
        if answer in out:
            return '(' + answer + ')'
    
    for answer in ["0", "1", "2", "3", "4"]:
        if answer + ':' in out:
            return '(' + chr(ord('A') + int(answer)) + ')'
        # if answer + ')' in out:
        #     return '(' + chr(ord('A') + int(answer)) + ')'
    
    return None
    
    

def map_prediction_to_answer_v2(row):
    answer_column = None
    if isinstance(row["pred"], str):
        prediction_letter = row["pred"][0]
        if prediction_letter in ["A", "B", "C", "D", "E"]:
            answer_column = "a" + str(ord(prediction_letter) - ord("A"))
        if "answer is " in row["pred"]:
            row["pred"] = row["pred"][row["pred"].index("answer is") :]
        if "A:" in row["pred"] or "A)" in row["pred"]:
            answer_column = "a0"
        elif "B:" in row["pred"] or "B)" in row["pred"]:
            answer_column = "a1"
        elif "C:" in row["pred"] or "C)" in row["pred"]:
            answer_column = "a2"
        elif "D:" in row["pred"] or "D)" in row["pred"]:
            answer_column = "a3"
        elif "E:" in row["pred"] or "E)" in row["pred"]:
            answer_column = "a4"
    if answer_column in ["a0", "a1", "a2", "a3", "a4"]:
        return row[answer_column]
    elif answer_column:
        print(prediction_letter)
    return "None"
