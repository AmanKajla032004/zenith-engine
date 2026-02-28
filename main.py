"""Zenith v1.0 — CLI entry point."""

from zenith import analyze, generate_report

if __name__ == "__main__":
    result = analyze("test_data.json")
    print(generate_report(result))