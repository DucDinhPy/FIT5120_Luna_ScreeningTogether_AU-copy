import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError


DATASET_NAME = "rag_eval"
DEPARTMENT = "Australian Government Department of Health, Disability and Ageing"
BOOKING_PDF = f"How to get a Cervical Screening Test _ {DEPARTMENT}.pdf"
SELF_COLLECTION_PDF = f"Self-collection for cervical screening _ {DEPARTMENT}.pdf"
RESULTS_PDF = f"Understanding your Cervical Screening Test results _ {DEPARTMENT}.pdf"
PROVIDER_PDF = "national-cervical-screening-program-a-summary-guide-for-healthcare-providers_0.pdf"
HOME_PDF = "Home-based-self-collection-guide-2024-1-2026-edit.pdf"


def example(question: str, answer: str, topic: str, *sources: tuple[str, int]) -> dict:
    return {
        "inputs": {"question": question},
        "outputs": {
            "answer": answer,
            "sources": [
                {"file_name": file_name, "page": page}
                for file_name, page in sources
            ],
        },
        "metadata": {"topic": topic, "answerable": bool(sources)},
    }


# Use PDF page numbers starting at 1, as stored by the ingestion pipeline.
EXAMPLES = [
    example(
        "Where can I get a Cervical Screening Test apart from a GP clinic?",
        "Other options include community health centres, women's health centres, "
        "family planning clinics, sexual health clinics and Aboriginal health services.",
        "access",
        (BOOKING_PDF, 1),
    ),
    example(
        "How can I find a cervical screening provider if I don't have a regular GP?",
        "Use healthdirect's online service finder to locate a healthcare provider.",
        "access",
        (BOOKING_PDF, 2),
    ),
    example(
        "Tôi có thể yêu cầu nhân viên y tế nữ thực hiện xét nghiệm tầm soát cổ tử cung không?",
        "Có. Bạn có thể đề nghị được một nhân viên y tế nữ thực hiện xét nghiệm.",
        "access",
        (BOOKING_PDF, 2),
    ),
    example(
        "Apart from asking my healthcare provider, how can I check when I am due?",
        "Check the National Cancer Screening Register's Participant Portal through "
        "myGov, or contact the Register by phone.",
        "access",
        (BOOKING_PDF, 2),
    ),
    example(
        "Why might I pay a fee for cervical screening even with a Medicare rebate?",
        "A consultation or laboratory charge can exceed the Medicare rebate. Ask the "
        "clinic whether the consultation and pathology are bulk billed.",
        "cost",
        (BOOKING_PDF, 3),
        (BOOKING_PDF, 4),
    ),
    example(
        "How do the two cervical screening collection options differ?",
        "With self-collection, the participant uses a swab to take a vaginal sample. "
        "With clinician collection, the healthcare provider uses a speculum and takes "
        "a sample from the cervix.",
        "self_collection",
        (BOOKING_PDF, 2),
    ),
    example(
        "Is a self-collected vaginal sample as effective at detecting HPV as a sample "
        "collected by a clinician?",
        "Yes. The two collection methods are equally effective for detecting HPV.",
        "self_collection",
        (SELF_COLLECTION_PDF, 1),
    ),
    example(
        "Who oversees self-collection and tells the patient their results?",
        "The healthcare provider orders and oversees self-collection, then communicates "
        "the results and any follow-up requirements to the patient.",
        "self_collection",
        (SELF_COLLECTION_PDF, 1),
    ),
    example(
        "Why might a clinician need to take another sample after HPV is found on "
        "a self-collected swab?",
        "The self-collected vaginal swab tests for HPV only. If follow-up requires "
        "checking cervical cells for changes, a clinician can take a cervical sample "
        "for liquid-based cytology.",
        "self_collection",
        (SELF_COLLECTION_PDF, 3),
        (PROVIDER_PDF, 6),
    ),
    example(
        "I can't attend a clinic. Who can arrange for VCS Pathology to mail me a "
        "cervical screening kit?",
        "A healthcare practitioner who refers to VCS Pathology can request a mailed kit "
        "after a telehealth consultation if the patient is eligible and due or overdue.",
        "home_collection",
        (HOME_PDF, 1),
    ),
    example(
        "How does someone send their home-collected sample back to VCS Pathology?",
        "The kit includes a prepaid Express Post return envelope addressed to VCS "
        "Pathology and instructions for packaging and posting the sample.",
        "home_collection",
        (HOME_PDF, 1),
    ),
    example(
        "Where should a person record the date they took a sample with the VCS home kit?",
        "Record the collection date on both the specimen tube and the pathology "
        "request form.",
        "home_collection",
        (HOME_PDF, 1),
    ),
    example(
        "What does a report saying 'return to screen in 5 years' mean?",
        "The test did not detect HPV and the next routine screening test is due in "
        "five years.",
        "results",
        (RESULTS_PDF, 1),
    ),
    example(
        "What does an 'unsatisfactory' cervical screening result mean?",
        "The laboratory could not read the sample. The test needs to be repeated.",
        "results",
        (RESULTS_PDF, 2),
    ),
    example(
        "What is a colposcopy after a Cervical Screening Test?",
        "A colposcopy is a procedure in which a specialist examines the cervix more "
        "closely to investigate screening findings.",
        "results",
        (RESULTS_PDF, 2),
    ),
    example(
        "What does the National Cancer Screening Register do for cervical screening "
        "participants?",
        "It records screening information, including results, and sends invitations "
        "and reminders when screening is due.",
        "register",
        (RESULTS_PDF, 3),
        (PROVIDER_PDF, 18),
    ),
    example(
        "If I opt out of the Register for cervical screening, does that also opt me "
        "out of bowel screening?",
        "The opt-out applies only to cervical screening. Participation in the bowel "
        "screening program is unaffected.",
        "register",
        (PROVIDER_PDF, 18),
    ),
    example(
        "How many Cervical Screening Tests were performed in Victoria in calendar "
        "year 2018?",
        "The supplied PDFs lack a Victorian test count for calendar year 2018.",
        "unanswerable",
    ),
    example(
        "How much will my own clinic charge me in total for cervical screening and "
        "pathology?",
        "The supplied PDFs give general billing information. Ask your clinic to confirm "
        "its consultation and pathology fees for an exact total.",
        "unanswerable",
    ),
    example(
        "Which cervical screening clinics in Carlton are open at 7 pm this Friday?",
        "The supplied PDFs lack live clinic opening hours. Check clinics' current "
        "hours and availability directly.",
        "unanswerable",
    ),
]


def publish_dataset(client: Client) -> int:
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            DATASET_NAME,
            description="Cervical screening RAG questions grounded in the project's PDFs.",
        )
        existing_questions = []
    else:
        existing_questions = [
            item.inputs.get("question")
            for item in client.list_examples(dataset_id=dataset.id)
        ]

    new_examples = [
        item for item in EXAMPLES if item["inputs"]["question"] not in existing_questions
    ]
    if new_examples:
        client.create_examples(dataset_id=dataset.id, examples=new_examples)
    return len(new_examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the LangSmith rag_eval dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Print the dataset as JSON.")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(EXAMPLES, indent=2))
        return

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        parser.error("Set LANGSMITH_API_KEY in backend/.env or the environment.")

    added = publish_dataset(Client())
    print(f"{DATASET_NAME}: added {added} of {len(EXAMPLES)} examples.")


if __name__ == "__main__":
    main()
