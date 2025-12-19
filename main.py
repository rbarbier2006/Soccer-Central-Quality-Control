import argparse
from profiles import PROFILES
from pdf_report import create_pdf_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate Soccer Central survey PDF report (Players or Families)."
    )
    parser.add_argument("input", help="Path to input Excel file (.xlsx)")
    parser.add_argument(
        "--survey",
        choices=list(PROFILES.keys()),
        required=True,
        help="Which survey type to run",
    )
    parser.add_argument(
        "--cycle",
        default="Cycle",
        help='Cycle label, example: "Cycle 3"',
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output PDF path (default: input name with _report.pdf)",
    )

    args = parser.parse_args()

    profile = PROFILES[args.survey]
    output_path = create_pdf_report(
        input_path=args.input,
        cycle_label=args.cycle,
        profile=profile,
        output_path=args.output,
    )

    print(f"Done. PDF saved to: {output_path}")


if __name__ == "__main__":
    main()
