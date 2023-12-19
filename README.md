# ambigDetection
Embodied Ambiguity Detection for LLM-enabled Robotics

This repo contains work done for Yale CSPS 572: AI Foundation Models class.

This is a ambiguity detector for LLM-based robot action planner. Given an environment with objects and actions that the robot can do this
system generates a plan to fullfill the request of the user by generating step by step plans. As well as generating plans the system also 
generates associated word probabilities at the generation step of the respective word.

In order to run first you need to obtain a HuggingFace key and place it in the llm.py file.
After that you also need to obtain access to llama2-13b-chat-hf from the hugginface website.

After completing all necessary permissions you can run an experiment by running test_llm.py file
It will ask you for a user prompt you can ask whatever you would want the robot to do keeping in mind that
robot is in an ewnvironment with following objects and actions

Actions:
Pick, Pour, Place, Start, Stop, Open, Close
Objects:
cereal, egg, bread, milk, banana, apple, yogurt, oats, water, bagel, cream cheese, toaster, microwave, pan, stove, table, plate, bowl, spoon, knife

In order the change actions or objects you can simply edit the list in test_llm.py file
