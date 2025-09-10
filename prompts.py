PROMPT_EXTRACT_SENTENCES = """\
Accurately extract sentences in the given text that refer to the image having the given caption. Response in JSON format with the only key "sentences" and the extracted sentences list as value.
Caption: {caption}

Text: 
'''
{text}
'''
"""

SYSTEM_PROMPT_VERIFY_SELF_EXPLANATORY = """You are a linguistic expert who can determine if a sentence is self-explanatory, \
meaning the sentence contains all the information for reader to understand referring further context."""

PROMPT_VERIFY_SELF_EXPLANATORY = """You are provided with a sentence referring a figure in a paper. Assume that the reader is provided with the figure, is the sentence understandable when looking at the figure? 
Sentence: "{sentence}"."""

PROMPT_GENERATE_QUESTIONS = """\
You are provided with a caption and a text referring a figure. Although you are not provided with the figure, use the caption to guess the content of the figure and generate technical questions that information inferred from the text are the answers as if you are looking at the figure.
Below are requirements for generating questions:
    - Remove the figure number, but always referring to the figure in both answers and questions. 
    - Avoid quoting or referring to specific facts, terms, abbreviations, dates, numbers, old/new versions, previous/next models or names, as these may reveal the conversation is based on the text information, rather than \
the figure itself. Focus on the technical and visual aspects of the figure that can be inferred without further information.
    - Ensure that questions are diverse and cover a range of technical and visual aspects of the figure.
    - Ensure that each question has an answer which must be extracted from the provided text for reliability but must be similar to technical conversation.
    - Answer responsibly, avoiding overconfidence, only extract information from the text to answer.
    - The question should ask for details, instead of simply asking for general overview/description.

NOTICE: If answers are vague or unmeaningful like "The figure illustrate the detailed dimensions.", delete corresponding questions. The conversation must be left empty if there is no meaningful questions and answers.

Example 1: Detailed and specific text.
Caption: "Fig. 4. The CAD drawing of the drive system: at the first stage, the belt transmits the rotation of the DC motor to the main shaft; and at the second stage, two pulleys on the main shaft transfer the motion to the output pulleys connected to the track blocks. The wheelbase i1 is tuned with an adjusting screw and the motor housing is anchored to the vehicle hull.",
Text: "Referring to Fig. 4, the first stage consists of the Hoyford timing belt that connects the pulley on the Rockstone DC motor output shaft to the pulley on the main shaft."
Question:
"conversation": [
    {{
        "user": "What component connects the pulley on the DC motor output shaft to the pulley on the main shaft in the first stage of the drive system as shown in the figure?",
        "assistant": "The first stage consists of the timing belt that connects the pulley on the DC motor output shaft to the pulley on the main shaft."
    }}
]

Example 2: Insufficient or too generic text.
Caption: "Fig. 5. Soft, toy sized humanoid upper body dimensions and kinematics.",
Text: "Detailed dimensions can be found in Fig. 5."
Question:
"conversation": []

########
Caption: "{caption}"
Text: "{sentences}"
Question:
"""