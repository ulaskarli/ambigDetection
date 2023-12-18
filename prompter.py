import os, time

class Prompter():
    def __init__(self) -> None:
        self.system_prompt_path = "prompts/code_prompt.txt"
        self.test_prompt_path = "prompts/test_prompt.txt"
        self.plan_prompt_path = "prompts/plan_prompt.txt"
        self.virthome_prompt_path = "prompts/virthome_prompt.txt"
        self.sys_tokens = ["<<SYS>>\n","\n<</SYS>>\n\n"]
        self.inst_tokens = ["[INST]","[/INST]"]
        self.env_prompt_path = "prompts/env_prompt.txt"
        
        with open(self.system_prompt_path, "r") as f:
            self.sys_template = f.read()
        with open(self.test_prompt_path,"r") as f:
            self.test_template = f.read()
        with open(self.plan_prompt_path, "r") as f:
            self.plan_template = f.read()
        with open(self.env_prompt_path, "r") as f:
            self.env_template = f.read()
        with open(self.virthome_prompt_path, "r") as f:
            self.virthome_template = f.read()

        self.code_history = ""
        self.plan_history = ""

    def generate_code_prompt(self,user_instruction):
        #msg = '<s>'+self.sys_tokens[0]+self.sys_template+self.inst_tokens[0]+user_instruction+self.inst_tokens[1]
        msg = self.inst_tokens[0]+self.sys_template+user_instruction+self.inst_tokens[1]
        return msg

    def generate_test_prompt(self,user_instruction):
        msg = '<s>'+self.sys_tokens[0]+self.test_template+self.sys_tokens[1]+self.inst_tokens[0]+user_instruction+self.inst_tokens[1]
        return msg
    
    def generate_plan_prompt(self,user_instruction):
        if self.plan_history == "":
            #msg = '<s>'+self.sys_tokens[0]+self.plan_template+self.sys_tokens[1]+self.inst_tokens[0]+user_instruction+self.inst_tokens[1]
            #msg = self.plan_template+'<s>'+self.inst_tokens[0]+user_instruction+self.inst_tokens[1]
            #msg = self.inst_tokens[0]+self.plan_template+user_instruction+self.inst_tokens[1]
            msg = self.sys_tokens[0]+self.plan_template+self.sys_tokens[1]+self.inst_tokens[0]+user_instruction+self.inst_tokens[1]
            self.plan_history+=msg
        else:
            msg = self.plan_history+"\n"+'<s>'+self.inst_tokens[0]+"Provide the next step for this plan."+self.inst_tokens[1]
            self.plan_history=msg
        return msg
    
    def update_plan_history(self,model_out):
        self.plan_history+="\n"+model_out

    def add_env_prompt(self,user_instruction,object_list):
        prompt=self.env_template+"\n"
        for o in object_list:
            prompt+=o+', '
        prompt=prompt[:-2]
        return prompt+"\n"+user_instruction
    
    def add_action_prompt(self,user_instruction,action_list):
        prompt="\nActions:\n"
        for a in action_list:
            prompt+=a+', '
        prompt=prompt[:-2]
        return prompt+"\n"+user_instruction
    
    def generate_virthome_prompt(self,user_instruction):
        if self.plan_history == "":
            #msg = '<s>'+self.sys_tokens[0]+self.plan_template+self.sys_tokens[1]+self.inst_tokens[0]+user_instruction+self.inst_tokens[1]
            #msg = self.plan_template+'<s>'+self.inst_tokens[0]+user_instruction+self.inst_tokens[1]
            #msg = self.inst_tokens[0]+self.plan_template+user_instruction+self.inst_tokens[1]
            msg = self.sys_tokens[0]+self.virthome_template+self.sys_tokens[1]+self.inst_tokens[0]+user_instruction+self.inst_tokens[1]
            self.plan_history+=msg
        else:
            msg = self.plan_history+"\n"+'<s>'+self.inst_tokens[0]+"Provide the next step for this plan."+self.inst_tokens[1]
            self.plan_history=msg
        return msg
    
