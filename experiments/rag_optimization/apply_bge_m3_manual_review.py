"""Apply the documented manual hallucination audit to the BGE-M3 result CSV."""

import csv
from pathlib import Path


CSV_PATH = Path(__file__).resolve().parents[2] / "eval" / "results_bge_m3_chroma_mmr_500_strict.csv"

# O = unsupported/incorrect factual generation; X = grounded answer or safe refusal.
REVIEWS = {
    0: ("X", "Safe refusal; retrieved context lacks the CATPart extension."),
    1: ("X", "CATProduct answer is supported by the assembly-information context."),
    2: ("X", "Safe refusal; no direct CATDrawing evidence was retrieved."),
    3: ("X", "Final claim is supported, but the answer contains unnecessary contradictory draft text."),
    4: ("X", "Additional Body before Boolean operations is supported."),
    5: ("X", "Listed import formats are supported by the retrieved lecture."),
    6: ("X", "Assembly creation steps are supported by the retrieved lecture."),
    7: ("X", "Mesh-control limitation and solid-only scope are supported."),
    8: ("X", "Knowledge Advisor parameters and formulas are supported."),
    9: ("X", "Grounded but incomplete: only Coincidence and Contact are listed."),
    10: ("X", "Safe refusal; CATPart evidence was not retrieved."),
    11: ("X", "CATProduct answer is supported by the assembly-information context."),
    12: ("X", "CATDrawing extension is supported by the retrieved context."),
    13: ("X", "Purple sketch means over-constrained; directly supported."),
    14: ("X", "Insert -> New Body is directly supported, despite verbose formatting."),
    15: ("X", "Import-format claim is supported by the retrieved lecture."),
    16: ("X", "Final safe refusal; no unsupported tool name is asserted."),
    17: ("X", "Very limited mesh control and solid-only scope are directly supported."),
    18: ("X", "Knowledge Advisor parameter and relation capabilities are supported."),
    19: ("X", "Grounded but incomplete: only Coincidence and Contact are listed."),
    20: ("X", "Final CATPart claim is supported, although the answer contains unnecessary draft text."),
    21: ("X", "New Product creation is supported by the assembly-document context."),
    22: ("O", "States CATPart for a drawing and includes an invented example.com citation."),
    23: ("X", "Final over-constrained/purple claim is supported, despite confused formatting."),
    24: ("O", "Answers Open Body instead of the requested New Body command."),
    25: ("O", "Claims IGES/STEP are for exporting; this is not supported for the asked import use."),
    26: ("X", "Existing-component insertion steps are supported by the retrieved guide."),
    27: ("X", "Solid-only limitation is directly supported."),
    28: ("X", "Formulas and Relations branch claim is supported."),
    29: ("X", "Constraints Toolbar claim is supported."),
    30: ("X", "Safe refusal for an out-of-manual price question."),
    31: ("X", "Safe refusal for an out-of-manual recommendation question."),
    32: ("X", "Safe refusal for an out-of-manual PC-specification question."),
    33: ("X", "Safe refusal for an out-of-manual course-fee question."),
    34: ("X", "Safe refusal; the evaluation reference marks this as out of scope."),
    35: ("X", "Safe refusal for a request not answered by the manual."),
    36: ("X", "Safe refusal for a perfect-conversion request."),
    37: ("X", "Safe refusal for a future-release question."),
    38: ("X", "Safe refusal for a recommendation question."),
    39: ("X", "Safe refusal for an undocumented macro request."),
}


def main():
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
        fieldnames = source_fieldnames = list(rows[0].keys())

    if len(rows) != 40 or set(range(len(rows))) != set(REVIEWS):
        raise ValueError("Expected exactly 40 review mappings for the BGE-M3 result CSV.")

    for index, row in enumerate(rows):
        flag, note = REVIEWS[index]
        row["hallucination_flag"] = flag
        row["reviewer_note"] = note

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
