# main.py

import argparse
import os

from pdf_report import create_pdf_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate survey PDF reports for players and/or families."
    )

    parser.add_argument(
        "input",
        help="Path to input Excel file (raw survey export).",
    )

    parser.add_argument(
        "--type",
        choices=["players", "families", "both"],
        default="players",
        help="Which survey profile to use.",
    )

    parser.add_argument(
        "--cycle",
        default="Cycle",
        help='Cycle label used in the PDF title (example: "Cycle 3").',
    )

    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output PDF path. If --type=both, this becomes a prefix and two PDFs "
            "will be created: <prefix>_players_report.pdf and <prefix>_families_report.pdf"
        ),
    )

    args = parser.parse_args()

    input_path = args.input
    cycle_label = args.cycle

    if args.type in ("players", "families"):
        out_path = args.output
        if out_path is None:
            # default handled inside create_pdf_report
            out_path = None

        pdf_path = create_pdf_report(
            input_path=input_path,
            cycle_label=cycle_label,
            survey_type=args.type,
            output_path=out_path,
        )
        print(f"Done. PDF saved to: {pdf_path}")
        return

    # args.type == "both"
    if args.output is None:
        base, _ = os.path.splitext(input_path)
        prefix = base
    else:
        # treat as prefix even if user passes .pdf
        prefix, ext = os.path.splitext(args.output)
        if ext.lower() == ".pdf":
            # user gave a pdf name; use it as prefix without .pdf
            prefix = prefix

    players_out = f"{prefix}_players_report.pdf"
    families_out = f"{prefix}_families_report.pdf"

    players_pdf = create_pdf_report(
        input_path=input_path,
        cycle_label=cycle_label,
        survey_type="players",
        output_path=players_out,
    )
    print(f"Players PDF saved to: {players_pdf}")

    families_pdf = create_pdf_report(
        input_path=input_path,
        cycle_label=cycle_label,
        survey_type="families",
        output_path=families_out,
    )
    print(f"Families PDF saved to: {families_pdf}")


if __name__ == "__main__":
    main()
