import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_STRACING"] = "true"


client = Client()

dataset_name = "chatbot_eval"

dataset = client.create_dataset(dataset_name)

client.create_examples(
    dataset_id = dataset.id,
    examples=[
        {
            "inputs": {"message": "What is cervical screening?"},
            "outputs": {
                "answer": "Cervical screening checks for HPV that can cause changes to cervical cells."
            },
        },
        {
            "inputs": {"message": "Where can I find a GP?"},
            "outputs": {
                "answer": "Use GP Finder and enter your Australian postcode.",
                "sources": [{"url": "/gpfinder"}],
            },
        },
        {
            "inputs": {
                "message": "How many people took a cervical screening test in Victoria in 2018?"
            },
            "outputs": {
                "answer": (
                    "The dataset has no separate count for Victoria in 2018. "
                    "It reports an 81.4% participation rate for the 2018–2022 window."
                ),
                "sources": [{"url": "/data"}],
            },
        },
        {
            "inputs": {
                "message": "Where do I open that?",
                "history": [{
                    "question": "Can I practise calling a clinic?",
                    "answer": "You can use Call Practice to rehearse a phone call.",
                }],
            },
            "outputs": {
                "answer": "Open Call Practice.",
                "sources": [{"url": "/call-practice"}],
            },
        },
    ],
)
