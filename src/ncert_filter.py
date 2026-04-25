"""Hard filtering for NCERT Subject and Class metadata."""

from __future__ import annotations
import pandas as pd

def filter_by_metadata(
    semantic_results: pd.DataFrame,
    target_subject: str | None = None,
    target_class: str | None = None,
    k: int = 7,
) -> pd.DataFrame:
    """
    Filters the FAISS semantic search results to strictly match 
    the user's selected Subject and Class.
    """
    if semantic_results.empty:
        return semantic_results

    filtered_df = semantic_results.copy()

    # Apply strict metadata filters
    if target_subject and target_subject != "All":
        filtered_df = filtered_df[filtered_df["subject"].astype(str).str.lower().str.startswith(str(target_subject).lower())]

    if target_class and target_class != "All":
        filtered_df = filtered_df[filtered_df["class"].astype(str) == str(target_class)]

    # Return the top K results after filtering
    return filtered_df.head(k).reset_index(drop=True)