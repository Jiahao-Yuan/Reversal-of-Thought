<h1 align="center"> Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up </h1>

RoT improves reasoning accuracy and efficiency while minimizing computational costs, leveraging **Preference-Guided Reverse Reasoning** and a **Cognitive Preference Manager** to optimally explore LLM reasoning with cognitive preferences.


## Model Architecture
<img src="./asset/Reversal_Of_Thought.png">

## Demo for _Preference-Guided Reverse Reasoning_
🎉🎉🎉 reversal_demo.py
```python
from utils.llm_utils import *
from utils.prompt import *
pipeline=Pipeline(model_id=model_id, base_url=base_url, api_key=api_key, prob=True)
demos = "Input:... Output:..." #Suggest 2-shot Demos
llm_taste=rot_pipeline( pipeline, reversal_of_thought, demos=demos, warmup=5)
```
## What might reversal_demo.py be used for?
- **Enhance LLM-Preferred Prompts for Task Solutions**  
  Refines prompts to align with LLM-preferred strategies, optimizing task-solving efficiency.

- **Potential for Creating Diverse QA Datasets**  
  Generates varied question-answer pairs to improve dataset diversity.
## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```bibtex
@article{yuan2024reversal,
  title={Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up},
  author={Yuan, Jiahao and Du, Dehui and Zhang, Hao and Di, Zixiang and Naseem, Usman},
  journal={arXiv preprint arXiv:2410.12323},
  year={2024}
}
```
