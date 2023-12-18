import re
from llm import LLM
from prompter import Prompter
import numpy as np
import matplotlib.pyplot as plt

object_list = ['cereal', 'egg', 'bread', 'milk', 
               'banana', 'apple', 'yogurt', 'oats', 'water', 
               'bagel', 'cream cheese','toaster','microwave',
               'pan','stove','table','plate', 'bowl', 'spoon', 'knife']

action_list = ['Pick', 'Pour', 'Place', 'Start','Stop', 'Open', 'Close']

v_action_list = ['walk','run','sit','standup','grab','open','close',
                 'putback','putin','switchon','switchoff','drink','touch','lookat']

v_object_list = ['apple','bananas','bellpepper','bread','bread_slice','carrot','coffee_pot','chefknife',
                 'chicken','chocolatesyrup','coffeemaker','cookingpot','candybar','crackers','cereal',
                 'chips','cucumber','fryingpan','milk','microwave','mug','salad','stove', 'fridge', 'diningtable']

def find_subarray(arr, subarr):
    s = []
    for i in range(len(arr) - len(subarr) + 1):
        if (arr[i:i+len(subarr)] == subarr).all():
            s.append(i)
    return s

def get_words_ids(gen,llm):
    gen = gen.replace("."," ")
    plan = re.split("\[[A-Za-z]+\s[0-9]+\]", gen)[1]
    return (llm.tokenize_object_space(plan.split()), plan.split())

def plan_step_analyzer():
    prompter = Prompter()
    foundation=LLM(model="llama",llama_size="13b")
    user_inst = input("Enter your input request: ")
    env_prompt = prompter.add_env_prompt(user_inst,object_list)#"Prepare breakfast."
    user_prompt = prompter.add_action_prompt(env_prompt,action_list)
    
    object_dict = foundation.tokenize_object_space(object_list)
    print(object_dict)
    action_dict = foundation.tokenize_object_space(action_list)
    print(action_dict)

    for i in range(5):
        llama_prompt = prompter.generate_plan_prompt(user_prompt)
        print(llama_prompt)
        outs = foundation.forward_pass(llama_prompt)

        gen = outs[0].sequences[:, outs[1]:][0].cpu().numpy()
        print(gen)
        decoded_model_out = foundation.decode_single_token(gen[:-1])
        print("Llama2->"+decoded_model_out)
        word_ids_return = get_words_ids(decoded_model_out,foundation)
        word_ids = word_ids_return[0]
        split_gen = word_ids_return[1]
        print(word_ids)
        if not len(decoded_model_out)>0:
            break
        prompter.update_plan_history(decoded_model_out.strip()+'</s>')

        token_hotspots = []
        word_probs = []
        for k in split_gen:
            id = word_ids[k]
            loc_scores = 0
            word = 0
            i = 0
            pos = find_subarray(gen[i:],id)
            ij = 0
            if len(pos) > 0:
                for i in pos:
                    loc_scores += outs[0].scores[i].cpu().numpy()[0]
                    loc = np.exp(outs[0].scores[i].cpu().numpy()[0])/sum(np.exp(outs[0].scores[i].cpu().numpy()[0]))
                    word += loc[id[ij]]
                    if len(id)>1:
                        ij+=1
                    else:
                        break
            token_hotspots.append(np.exp(loc_scores)/sum(np.exp(loc_scores)))
            word_probs.append(word)
        print(word_probs)

        # k=1
        # m = [m for m in range(len(token_hotspots[0]))]
        # for place in token_hotspots:
        #     plt.plot(m, place)
        #     plt.title("Probability of word: "+str(k)+" over vocab ")
        #     plt.show()
        #     k+=1

        l = [l for l in range(len(split_gen))]
        f, ax = plt.subplots()
        ax.plot(l, word_probs)
        ax.set_xticks(l)
        ax.set_xticklabels(split_gen)
        plt.title("Probability of each word")
        plt.xticks(rotation=90)
        plt.show()

def test_single():
    prompter = Prompter()
    foundation=LLM(model="llama",llama_size="13b")

    env_prompt = prompter.add_env_prompt("cut a slice of bread for me.",object_list)
    user_prompt = prompter.add_action_prompt(env_prompt,action_list)
    
    
    object_dict = foundation.tokenize_object_space(object_list)
    print(object_dict)
    action_dict = foundation.tokenize_object_space(action_list)
    print(action_dict)

    for i in range(3):
        llama_prompt = prompter.generate_plan_prompt(user_prompt)
        print(llama_prompt)
        outs = foundation.forward_pass(llama_prompt)

        gen = outs[0].sequences[:, outs[1]:][0].cpu().numpy()
        print(gen)
        decoded_model_out = foundation.decode_single_token(gen[:-1])
        print("Llama2->"+decoded_model_out)
        print(get_words_ids(decoded_model_out,foundation))
        if not len(decoded_model_out)>0:
            break
        prompter.update_plan_history(decoded_model_out+"</s>")

        gen_object_id = []
        for id in action_dict.values():
            pos = find_subarray(gen,id)
            if len(pos) > 0:
                for i in pos:
                    gen_object_id.append(i)
        for id in object_dict.values():
            pos = find_subarray(gen,id)
            for i in pos:
                    gen_object_id.append(i)
        gen_object_id.sort()
        print(gen_object_id)

        token_hotspots = []
        for id in gen_object_id:
            loc_scores = outs[0].scores[id].cpu().numpy()[0]
            token_hotspots.append(np.exp(loc_scores)/sum(np.exp(loc_scores)))

        k=0
        m = [m for m in range(len(token_hotspots[0]))]
        for place in token_hotspots:
            
            probs = {}
            for obj in object_list:
                t = object_dict[obj]
                probs[obj]=place[t[0]]

            probs_act = {}
            for act in action_list:
                t = action_dict[act]
                probs_act[act]=place[t[0]]

            
            if sum(probs.values())>0:
                print(probs)
                plt.bar(probs.keys(), probs.values())
                plt.title("Token Probabilities")
                plt.xticks(rotation=90)
                plt.show()
            elif sum(probs_act.values())>0:
                print(probs_act)
                plt.bar(probs_act.keys(), probs_act.values())
                plt.title("Token Probabilities")
                plt.xticks(rotation=90)
                plt.show()

            plt.plot(m, place)
            plt.title("Probability over Vocab at "+str(gen_object_id[k]))
            plt.show()
            k+=1

def virthome_step_analyzer():
    prompter = Prompter()
    foundation=LLM(model="llama",llama_size="13b")

    env_prompt = prompter.add_env_prompt("Prepare a healthy meal.",v_object_list)
    user_prompt = prompter.add_action_prompt(env_prompt,v_action_list)
    
    object_dict = foundation.tokenize_object_space(v_object_list)
    print(object_dict)
    action_dict = foundation.tokenize_object_space(v_action_list)
    print(action_dict)

    for i in range(10):
        llama_prompt = prompter.generate_virthome_prompt(user_prompt)
        print(llama_prompt)
        outs = foundation.forward_pass(llama_prompt)

        gen = outs[0].sequences[:, outs[1]:][0].cpu().numpy()
        print(gen)
        decoded_model_out = foundation.decode_single_token(gen[:-1])
        print("Llama2->"+decoded_model_out)
        word_ids_return = get_words_ids(decoded_model_out,foundation)
        word_ids = word_ids_return[0]
        split_gen = word_ids_return[1]
        print(word_ids)
        if not len(decoded_model_out)>0:
            break
        prompter.update_plan_history(decoded_model_out.strip()+'</s>')

        token_hotspots = []
        word_probs = []
        for k in split_gen:
            id = word_ids[k]
            loc_scores = 0
            word = 0
            i = 0
            pos = find_subarray(gen[i:],id)
            ij = 0
            if len(pos) > 0:
                for i in pos:
                    loc_scores += outs[0].scores[i].cpu().numpy()[0]
                    loc = np.exp(outs[0].scores[i].cpu().numpy()[0])/sum(np.exp(outs[0].scores[i].cpu().numpy()[0]))
                    word += loc[id[ij]]
                    if len(id)>1:
                        ij+=1
                    else:
                        break
            token_hotspots.append(np.exp(loc_scores)/sum(np.exp(loc_scores)))
            word_probs.append(word)
        print(word_probs)

        # k=1
        # m = [m for m in range(len(token_hotspots[0]))]
        # for place in token_hotspots:
        #     plt.plot(m, place)
        #     plt.title("Probability of word: "+str(k)+" over vocab ")
        #     plt.show()
        #     k+=1

        l = [l for l in range(len(split_gen))]
        f, ax = plt.subplots()
        ax.plot(l, word_probs)
        ax.set_xticks(l)
        ax.set_xticklabels(split_gen)
        plt.title("Probability of each word")
        plt.xticks(rotation=90)
        plt.show()

if __name__ == '__main__':
    #test_single()
    plan_step_analyzer()
    #virthome_step_analyzer()