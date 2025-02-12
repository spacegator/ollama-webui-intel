
def format_results_as_markdown_table(results: dict[str, dict[str, float]])\
        -> str:
    """Formats the results as a Markdown table.

    Args:
        results (dict[str, dict[str, float]]): The results dictionary
            containing mean and STD tokens/s for each model.

    Returns:
        str: A formatted Markdown table.
    """
    headers = ["Model", "Mean ± STD Tokens/s"]
    rows = []

    for model, stats in results.items():
        row_content = (f"{stats['mean_tokens_per_second']:.2f} ± "
                       f"{stats['std_dev_tokens_per_second']:.2f}")
        rows.append([model, row_content])

    # Calculate the maximum width for each column
    max_widths: list[int] = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            max_widths[i] = max(max_widths[i], len(cell))

    markdown_table: str = "| " + " | ".join(
        header.ljust(width)
        for header, width in zip(headers, max_widths, strict=True)) \
        + " |\n"
    markdown_table += "|-" + \
        "-|-".join("-" * width for width in max_widths) + " |\n"

    for row in rows:
        markdown_table += "| " + \
            " | ".join(cell.ljust(width)
                       for cell, width in zip(row, max_widths, strict=True)) \
            + " |\n"

    return markdown_table
