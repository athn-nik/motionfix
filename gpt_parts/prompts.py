
from gpt_parts.joint_utils import coarse_bp, less_coarse_bp, less_coarse_bp_v2, less_coarse_bp_v3

cp_str = '\n'.join(coarse_bp)
lcp_str = '\n'.join(less_coarse_bp)
lcp_str_v2 = '\n'.join(less_coarse_bp_v2)
lcp_str_v3 = '\n'.join(less_coarse_bp_v3)

edit_prompts_to_gpt = "You will be given an edit text that is supposed to be used to edit a motion." \
  "Your task is given the text to determine what are the parts of the motion that should change based on that edit text." \
    "The instructions for this task are to " \
                      f"choose your answers from the list below:\n{lcp_str_v2}\n" \
                      "Here are some examples of the question and " \
                      "answer pairs for this task:\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: do it earlier?\n" \
  "Answer: right leg\nleft leg\nbuttocks\nleft arm\nright arm\ntorso\n neck\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: faster?\n" \
  "Answer: right leg\nleft leg\nbuttocks\nleft arm\nright arm\ntorso\n neck\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: do it with the opposite leg?\n" \
  "Answer: right leg\nleft leg\nbuttocks\n Question: What are the body parts that should be edited in the motion if the edit text is: raise your arm higher?\n" \
  "Answer: right arm\nleft arm\ntorso\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: raise your left arm higher?\n" \
  "Answer: left arm\ntorso\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: take the hands a bit more up while doing the jumping jacks?\n" \
  "Answer: right arm\nleft arm\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: start the action while being upright instead of getting up at the start\n" \
  "Answer: right leg\nleft leg\nbuttocks\nleft arm\nright arm\ntorso\n neck\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: don't bend to the side and keep hands separate at shoulder width\n" \
  "Answer: buttocks\nleft arm\nright arm\ntorso\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: stop in the end?\n" \
  "Answer: right leg\nleft leg\nbuttocks\nleft arm\nright arm\ntorso\n neck\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: do it one more time?\n" \
  "Answer: right leg\nleft leg\nbuttocks\nleft arm\nright arm\ntorso\n neck\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: mirror your motion?\n" \
  "Answer: right leg\nleft leg\nbuttocks\nleft arm\nright arm\ntorso\n neck\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: start earlier?\n" \
  "Answer: right leg\nleft leg\nbuttocks\nleft arm\nright arm\ntorso\n neck\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: stop moving in the end?\n" \
  "Answer: right leg\nleft leg\nbuttocks\nleft arm\nright arm\ntorso\n neck\n" \
  "Question: What are the body parts that should be edited in the motion if the edit text is: [EDIT TEXT]\n"



final_prompts_to_gpt = [
  f"The instructions for this task are to choose your answers from the list below:\n{lcp_str}\nHere are some examples of the question and answer pairs for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nglobal orientation\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nQuestion: What are the body parts involved in the action of: sit down?\nAnswer: right leg\nleft leg\nglobal orientation\nQuestion: What body parts are involved in the action of: [ACTION]?", 
                  f"The instructions for this task are to choose your answers from the list below:\n{lcp_str_v2}\nHere are some examples of the question and answer pairs for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nbuttocks\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nQuestion: What are the body parts involved in the action of: sit down?\nAnswer: right leg\nleft leg\nbuttocks\nQuestion: What body parts are involved in the action of: [ACTION]?",
                  f"The instructions for this task are to choose your answers from the list below:\n{lcp_str_v3}\nHere are some examples of the question and answer pairs for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nbuttocks\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nQuestion: What are the body parts involved in the action of: sit down?\nAnswer: right leg\nleft leg\nbuttocks\nwaist\nQuestion: What body parts are involved in the action of: [ACTION]?"]


prompts_to_gpt = [f"Be detailed and accurate, list the body parts involved in this action: [ACTION]\nPlease, use the body parts from this list:\n{lcp_str}",
                  
                  f"Please, use the body parts from this list:\n{lcp_str}\nBe detailed and accurate, list the body parts involved in this action: [ACTION]\n",
                  
                  f"Please, use the body parts from this list:\n{lcp_str}\nPlease list the body parts involved in this action: [ACTION]\n",
                  
                  "Be detailed and accurate, list the body parts involved in this action:[ACTION]",

                  "List the body parts involved in this action:[ACTION]",
                  
                  f"Choose from the following list the body parts that are involved in the action of: [ACTION] \n{lcp_str}",
                  

                  f"Choose from the following list only the body parts that are involved in the action of: [ACTION] \n{lcp_str}\nInclude body parts only from the list above in you answer.",
                  
                  f"Here are some examples of the questions and answers I want for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nglobal orientation\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nWhat body parts are involved in the action of:[ACTION]?",
                  
                  f"Here are some examples of the questions and answers pairs for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nglobal orientation\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nChoose your answer from this list only:\n{lcp_str}\nQuestion: What body parts are involved in the action of:[ACTION]?",
                
                f"The instructions for this task are to choose your answers from the list below:\n{lcp_str}\nHere are some examples of the question and answer pairs for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nglobal orientation\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nQuestion: What body parts are involved in the action of: [ACTION]?",
                
                f"The instructions for this task are to choose your answers from the list below:\n{lcp_str_v2}\nHere are some examples of the question and answer pairs for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nbuttocks\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nQuestion: What body parts are involved in the action of: [ACTION]?",
                
                f"The instructions for this task are to choose your answers from the list below:\n{lcp_str}\nHere are some examples of the question and answer pairs for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nglobal orientation\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nQuestion: What are the body parts involved in the action of: sit down?\nAnswer: right leg\nleft leg\nglobal orientation\nQuestion: What body parts are involved in the action of: [ACTION]?",
                
                f"The instructions for this task are to choose your answers from the list below:\n{lcp_str_v2}\nHere are some examples of the question and answer pairs for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nbuttocks\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nQuestion: What are the body parts involved in the action of: sit down?\nAnswer: right leg\nleft leg\nbuttocks\nQuestion: What body parts are involved in the action of: [ACTION]?",
                
                f"{lcp_str}\nChoose from the list above the body parts that are involved in the action of: [ACTION]",
                
                f"Choose only from the following list the body parts that are involved in the action of: [ACTION] ?\n{lcp_str}",
                f"The instructions for this task are to choose your answers from the list below:\n{lcp_str_v3}\nHere are some examples of the question and answer pairs for this task:\nQuestion: What are the body parts involved in the action of: walk forwards?\nAnswer: right leg\nleft leg\nbuttocks\nQuestion: What are the body parts involved in the action of: face to the left?\nAnswer: torso\nneck\nQuestion: What are the body parts involved in the action of: put headphones over ears?\nAnswer: right arm\nleft arm\nneck\nQuestion: What are the body parts involved in the action of: sit down?\nAnswer: right leg\nleft leg\nbuttocks\nwaist\nQuestion: What body parts are involved in the action of: [ACTION]?",
                ]
