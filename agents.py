from utils import *
from tools import *
from inference import *

class BaseAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, max_steps=100):
        if notes is None: self.notes = []
        else: self.notes = notes
        self.max_steps = max_steps
        self.model = model
        self.phases = []
        self.plan = str()
        self.report = str()
        self.history = list()
        self.prev_comm = str()
        self.prev_report = str()
        self.exp_results = str()
        self.dataset_code = str()
        self.model_code = str()
        self.results_code = str()
        self.prev_code = str()
        self.data_detail = str()
        self.paper_summary = str()
        self.data_summary = str()
        self.max_hist_len = 5

    def set_model_backbone(self, model):
        self.model = model

    @staticmethod
    def clean_text(text):
        """
        Fix minor corrections
        :return: (str) corrected text
        """
        text = text.replace("```\n", "```")
        return text

    def inference(self, instruction, phase, step, feedback="", info=None, image=None, temp=None, use_command=True):
        # context = self.context(phase)
        history_str = "\n".join([_[1] for _ in self.history])
        phase_notes = [_note["note"] for _note in self.notes if phase in _note["phases"]]
        notes_str = f"Notes for the task objective: {phase_notes}\n" if phase_notes else ""
        if use_command == True:
            sys_prompt = f"""You are {self.role_description()}"""
            prompt = (
                f"""{'~' * 10}\nHistory: {history_str}\n{'~' * 10}\n\n""" 
                f"Current Step #{step}, Phase: {phase}\n"
                f"Task instructions:{self.phase_prompt(phase)}\n"
                f"[Overall Objective] Your oveall goal is to follow the instruction to replicate the method proposed in the paper. Instruction: {instruction}."
                "To achieve the objective, start by conducting literature review, learning relevant codes, and finally generating the method and experiment codes.\n"
                f"{info}\n{feedback}\n{self.command_descriptions(phase)}\n{notes_str}\n."
                "When you are given commands that you can use, your reply must be selected from among the commands.\nPlease produce a single command below:\n")
        else:
            sys_prompt = f"""You are {self.role_description()}"""
            prompt = (
                f"Current Step #{step}, Phase: {phase}\n"
                f"[Overall Objective] Your oveall goal is to follow the instruction to replicate the method proposed in the paper. Instruction: {instruction}."
                f"To achieve the objective, start by conducting literature review, learning relevant codes, and finally generating the method and experiment codes."
                f"{info}\nTask instructions: {self.phase_prompt(phase)}{notes_str}")
        if image == None:
            model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, temp=temp)
        else:
            model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=(prompt, image), temp=temp)
        print("^"*50, phase, "^"*50)
        model_resp = self.clean_text(model_resp)
        self.prev_comm = model_resp
        steps_exp = None
        if feedback is not None and "```EXPIRATION" in feedback:
            steps_exp = int(feedback.split("\n")[0].replace("```EXPIRATION ", ""))
            feedback = extract_prompt(feedback, "EXPIRATION")
        self.history.append((steps_exp, f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Your response: {model_resp}"))
        # remove histories that have expiration dates
        for _i in reversed(range(len(self.history))):
            if self.history[_i][0] is not None:
                self.history[_i] = self.history[_i] = self.history[_i][0] - 1, self.history[_i][1]
                if self.history[_i][0] < 0:
                    self.history.pop(_i)
        if len(self.history) >= self.max_hist_len:
            self.history.pop(0)
        return model_resp

    def reset(self):
        self.history.clear()  # Clear the deque
        self.prev_comm = ""

    def phase_prompt(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def role_description(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def command_descriptions(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")

    def example_command(self, phase):
        raise NotImplementedError("Subclasses should implement this method.")


class ResearchAgent(BaseAgent):
    def __init__(self, model="gpt4omini", notes=None, max_steps=100, task_instruction=None):
        super().__init__(model, notes, max_steps)
        self.phases = [
            "overall summary",
            "method summary",
            "experiment summary",
            "paper lineage",
            "data acquisition",
            "method replication",
            "experiment execution",
            "update"
        ]
        self.paper = []
        self.task_instruction = task_instruction

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "paper lineage":
            return (
                "Add the most related work after reading using the following command: ```ADD\n<related work list>\n```\nwhere <related work list> is the list of related work list in the renference, the item in the list should be the full name of the paper. ADD is just the word ADD.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. ADD).\n"
                )
        elif phase == "method replication":
            return (
                "If the code-summary comparison is not well aligned, you should update the method summary using the command: ```UPDATE\n<new_summary>```\nwhere <code-summary> needs to emphasize the mismatched contents in the code-summary comparison and new summarized content. UPDATE is just the word UPDATE."
                "If the code-summary comparison is completely aligned, you can submit code with the command: ```SUBMIT\n```\n. The SUBMIT is just the word SUBMIT.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT, UPDATE).\n"
            ) 
        elif phase == "experiment execution":
            return (
                "If the code-summary comparison is not well aligned, you should update the method summary using the command: ```UPDATE\n<update_summary>\n``` needs to emphasize the mismatched contents in the code-summary comparison. UPDATE is just the word UPDATE."
                "If the code-summary comparison is completely aligned, you can submit code with the command: ```SUBMIT\n```\n. The SUBMIT is just the word SUBMIT.\n"
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. SUBMIT, UPDATE).\n"
            )             
        else:
            return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")

        if phase == "overall summary":
            phase_str = (
                "Provide a comprehensive summary of the given paper including the method and experimental setting. Provide an overview that should include the proposed core method, the mechanism by which the method addresses the problem, the main performance metrics, as well as a brief description of the experimental setup and the comparison benchmark."
                "Subsequently, offer a detailed exposition of the critical elements for replication, emphasizing their interdependencies. This should include the detailed model architecture, the experimental configuration, the mathematical formulations and algorithms, the training protocol. Finally, explain how all the contents in the paper should be reproduced."
                "If the paper references a GitHub repository, provide the link using markdown in the format [name](link). Ensure the summary intricately elucidates the relationships between these components to enable precise reproduction, moving beyond a mere enumeration of individual aspects."
                "Note that when you summarize a formula, each parameter within the formula must have a corresponding explanation. If the parameters come from other formulas, you also need to explain these parameters."
                "The overall reproduction will be devided into three phase, the data acquisition, the method replication and the experiment running. You need to summarize all the details of paper into these three phase and then formulate plans about what should be implemented in these three phases. In your plan, you need to clearly describe the specific details of each model and experiment, rather than using etc."
            )
        elif phase == "method summary":
            return (
                "You are going to summarize the methods in the paper, including details on the proposed model architecture, formulas, and algorithms. The formulas and algorithms should in the latex format."
                "You need to focus mainly on the model structure, specifically detailing what layers used in the model. Do not summarize anything else except methods. If there are detailed formulas and algorithms in the paper, you need to extract formulas with the corresponding each layer or module of the model, and extract the overall algorithm of the paper at the end of the summary."
                f"Your task is to reproduce the {self.task_instruction}. Therefore, you must clearly state all the method hyperparameters such as the number of layers of the model, the model dimension, and other method related parameters, according to the task, model and metrics in the intructions."
                "Don't only generate formulas, summarize the overall process as well. Note that when you summarize a formula, each parameter within the formula must have a corresponding explanation."
                "The important point is that the method you have summarized should be directly applicable for conducting experiments."
                "After summarization, you need to give a plan to describe how to reproduce the experiment mentioned in the instruction. For example, how to reproduce the proposed method, how to combine the reproduced method in the exisiting pretrained model, what the parameters should be set. All the pretrained models, model structures should be reprodeuced."
            )
        elif phase == "experiment summary":
            return (
                f"You are going to summarize the experiment in the paper, including details on trianing strategies, hyperparameters and settings."
                "As the previous method reviews mainly cover the contents related to model architectures, you still need to summarize information beyond the model structures, such as those concerning loss functions, evaluation metrics and so on."
                f"Your task is to reproduce the {self.task_instruction}. You need to summarize all the experiment details, such as the training strategies , epoches, optimizer, learning rate, and other hyperparameters."
                "At the same time, you also need to pay attention to the parameter settings in the appendix. Many parameters, such as the specific details of learning rate decay, will be described in the appendix."
                "After summarization, you need to give a plan to describe how to reproduce the experiment mentioned in the instruction, including traing, evaluating and testing. For example, the training, evalating and testing pipline, the evaluation metrics, the details about training such as parameters update strategies, the parameters about different dataset. All the experiment seting, parameter update strategy should be reprodeuced."
            )
        elif phase == "paper lineage":
            phase_str = (
                "Your task is to read the paper and identify the 3 most relevant papers from its references that help in understanding the paper's contributions, including the proposed model architecture, experimental settings, and other details. These papers need to be in the same research field as the ones that need to be replicated."
                "You need to infer the most relevant related works based on the information such as the position and name of the reference. Imoratantly, the name of the paper should be correct, do not generate mismatched name."
                "The selected papers must come from the references and be specific to the same research field as the paper, avoiding commonly cited works like 'Attention Is All You Need'. Return the related works in the format: ['paper name 1', 'paper name 2', ...] with only paper names (author names should not be included)."
                "You should search the paper from then references section, do not generate name by yourself. Also do not add the paper in the instructions."
            )
        elif phase == 'data acquisition':
            phase_str = (
                "Your goal is review the code and output a summary of code details and instructions about how to use th code. In addition, you need to analyze the specific meaning of each dimension of the data, such as which dimension represents batch size, which dimension represents time, which dimension represents image height, and which dimension represents features."
            )
        elif phase == "method replication":
            phase_str = (
                "You are going to compare the model structure and data shape in the generated code with the method described in the summary step by step. Determine if the code matches the summary in detail, ensuring every novel contribution corresponds to specific model layers, loss function or data dimensions. You just need to compare the model architecture, not the training settings."
                "The number of layers in the comparison model is of great significance. In your method summary, it is often included that the number of layers of the model and the settings of each scale, where each scale represents one layer. You need to determine whether the model has generated the corresponding multi-layer structure."
                "After comparison, if the code does not match the summary, you need to inform to update the summary. Analysis the mismatched content and inform which part should be emphasized when update the summay."
                "To avoid any oversight, please update the summary several times before submitting to ensure its accuracy. Do not submit code too early."
            )
        elif phase == "experiment execution":
            phase_str = (
                "Compare the paper's training settings with the implemented code. Identify discrepancies in parameters (loss functions, data dimensions, hyperparameters) and optimization methods (learning rate schedules, gradient decay) step by step. Every item in the summary needs to correspond to the code."
                "After comparison, If the comparison shows code do not matches the summary, update the summary by rereading the original paper, current summary, and the code-summary comparison."
                "If the comparison shows code fully matches the summary with no updates needed you could consider SUBMIT the code. To avoid any oversight, please update the summary several times before submission to ensure its accuracy. Importantly, only the comparison fully matches then you could submit."
                "You do not need to compare the model architecture, as the model architecture is well aligned. Also, the trianning epoch and batchsize are defined for customization, do not compare them with setting in the summary."
            )
        elif phase == "update":
            phase_str = (
                "You need to update the previous summary now. You have been provided with the full text, the previous summary, as well as the comparison between the code and the summary. You need to re-read the full text based on this comparison and update the previous summary to include the contents that were not covered before."
                "The new summary should focus on: 1. mismatches identified in the code-summary comparison, and 2. content mentioned in the paper but missing from the summary, emphasizing these points. New content should be appended to the existing summary without repeating previous summary content."
            )
        else:
            phase_str = ""
        return phase_str

    def role_description(self):
        return "Your overall goal is to reproduce the code in the paper based on the paper and the instructions. You are a review expert that need to summarize the paper key points and compare the code with the paper. "

    def add_review(self, full_text, review_text):
        try:
            review_entry = {
                "full_text": full_text,
                "summary": review_text,
                "method_summary": '',
                "experiment_summary": ''
            }
            self.paper.append(review_entry)
            return f"Successfully added paper ", full_text
        except Exception as e:
            return f"Error trying to add review -- bad formatting, try again: {str(e)}", ""

    def update_review(self, update_content, section):
        try:
            update_content = update_content.strip()
            if section == 'method':
                self.paper[0]['method_summary'] += f'\n{update_content}'
            elif section == 'experiment':
                self.paper[0]['experiment_summary'] += f'\n{update_content}'
            return f"Successfully upated {section} summary", update_content
        except Exception as e:
            return f"FAIL to update", ""


class CodeAgent(BaseAgent):
    def __init__(self, model="o1-preview", notes=None, max_steps=100, dataloader_code=None, task_instruction=None):
        super().__init__(model, notes, max_steps)
        self.phases = [
            "paper lineage",
            "data acquisition",
            "method replication",
            "experiment execution",
            "code refactoring",
            "error analysis"
        ]
        self.code = []

        self.dataloader_code = dataloader_code
        self.task_instruction = task_instruction

    def command_descriptions(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase == "paper lineage":
            return (
                "You could select related code file using the following command: ```ADD\n<related code file>\n```\nwhere <related code file> is the list of related code file in given code repo, the item in the list should be the full name file path. ADD is just the word ADD.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. ADD).\n"
                )
        elif phase == "data acquisition" or phase == "method replication" or phase == "experiment execution":
            return (
                "You can edit code using the following command: ```EDIT N M\n<new code>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the the last line index you want to replace (everything inbetween will also be removed), and <new code> will be the new code that is replacing the old code."
                "This command allows you to replace lines indexed n through m (n:m) of the current code with as many lines of new code as you want to add. This will be the primary way that you interact with code. \n"                
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. EDIT).\n"
            )
        return ""

    def phase_prompt(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        if phase =='paper lineage':
            phase_str = (
               f"You need to reproduce the {self.task_instruction} experiment now. The code repository mentioned above is related to the target repriduction paper. Please filter out the code files that are helpful for the reproduction based on the instructions, and return the useful code files in the form of a list. For example, ```python ['file1.py', 'model/file2.py']```. You should start from the most relevant code file."
                )
        elif phase == "data acquisition":
            phase_str = (
                "If you are not given a dataloader code, you need to generate them from scratch. You should analyze the dataset mentioned in the instruction and generate the corresponding dataloade code. You need to generate the code to download the dataset and corresponding dataloader."
                "You need to generate a code snippet to check the shape, type of data in the first batch of dataloader. Modify the code to fix bugs, ensuring modifications the existing code. Analyze and comment on the meaning of each data dimension (e.g., batch size, time, image height, features) for clarity."
                )
        elif phase == "method replication":
            phase_str = (
                "You need to implement the code of model architecture based on the provided method summary, fixing any code errors. You are suggested to establish a complete framework firstly. All the network structures you generate must have specific implementations. Importantly, Ensure the model's input and output align with the dataloader by testing with data in the dataloader (e.g., the first batch) to verify dimensions."
                "Modify the code to fix bugs or meet requirements. If the generated code match an existing snippet, edit the existing code instead of adding new snippets. Use prior summaries and code from related work in the same field to guide implementation, as model structures are often similar."
                "Please note that you should avoid the situation where you only implement the layer of the model but neglect the issue of the number of layers. You should not merely replicate the implementation of the innovative layers of the model but ignore the number of layers."
                "All the parameters should be defined keyword arguments. You should reproduce the method utilized in the paper for the given experiment. When the paper contains pretrained model, do not utilize generate dummy model, you should correctly utilize the model."
                )
        elif phase == "experiment execution":
            phase_str = (
                "You need implement the code of experiments in the paper according to the summary provided to you and fix any error produced by the code."
                "Define a main function and call other functions to complete the experiment."
                "During the process of fixing bugs, it is better to edit a complete code snippet, such as a function or a class, to avoid deleting other functions or encountering IndentationError."
                "Your main goal is to complete the entire experiment code. Generate the full experiment process (trainging and testing) and break at the first batch finish for debugging. Epoch related setting, such as the learning rate decay mentioned in the summary, should be added."
                "Break the training validation and testing process at the first batch at the first epoch for debugging. You only need to use ```break``` or ```continue`` at the appropriate places. Do not set ```epoch``` to 1."
                "Do not modify the model architecture code unless absolutely compelled, because the model architecture has been well aligned already."
                )
        elif phase == "error analysis":
            return (
                "The previously generated code is in error. You need to analyze the cause of the error now and guide the subsequent code modification."
                "There is no need to generate a too lengthy analysis. Focus on locate the position of the error and the cause of the error. Give suggestions about how to fix them."
                "For loading pretrained checkpoints, please inform how to modify the model architecture code to load the checkpoints."
                )
        elif phase == "code refactoring":
            return (
                "All the experiment code have been completed now. However, due to the previous modifications, there are some code redundancy and repeated definitions that do not conform to the standards. Please now standardize the provided codes."
                "Previously, in order to quickly debug, some training and testing were carried out within loops and then broken out. These parts were removed to achieve a complete experimental code."
                "Do not modify the settings and function name in the given code. Just refactor the previous generated code."
                )
        else:
            phase_str = ""
        return phase_str

    def role_description(self):
        return "Your overall goal is to reproduce the code in the paper based on the paper and the instructions. You are a code expert that need to reproduce the experiment code and fix the bugs in the code."

    def add_code(self, related_repo, review):
        try:
            entry_code = {
                "related_code": related_repo,
                "review": review
            }
            self.code.append(entry_code)
            return f"Successfully added code repo", related_repo
        except Exception as e:
            return f"Error trying to add review -- bad formatting, try again: {str(e)}", ""

    def format_relate_repo(self, related_repo):
        instruct = "There are some related code repo that contains many common used model architectures and experimental setups that are not mentioned in the paper but which are highly relevant to the provided results.  \
                    You need to learn some common model structures and training tricks related to this from it. Also the comparison of the paper of these code with current replicating paper is provided." \
                    "For example, the paper may use the attention method in the given related repo rather than multi-head attention, so you could replicate paper based on the repo attention mechanism."
        return "\n".join(related_repo)+"\n"+instruct
