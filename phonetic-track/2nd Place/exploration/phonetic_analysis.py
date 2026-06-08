import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import jiwer
from pathlib import Path

# --- Your Starting Variables ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPLORATION_DIR = PROJECT_ROOT / "exploration"
PARQUET_FILE = EXPLORATION_DIR / "engineered_audio_features.parquet"
WORD_FILE = DATA_DIR / "train_word_transcripts.jsonl"

def get_phoneme_counts(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """Extracts and counts individual phonemes, ignoring spaces and stress marks."""
    # Remove spaces and primary/secondary stress marks for a cleaner count
    clean_str = pl.col(col_name).str.replace_all(r"[ ˈˌ]", "")
    
    return (
        df.select(clean_str.str.split(""))
        .explode(col_name)
        .filter(pl.col(col_name) != "")
        .group_by(col_name)
        .len(name="count")
        .sort("count", descending=True)
    )

def plot_distributions(proper_counts: pl.DataFrame, actual_counts: pl.DataFrame):
    """Plots interactive bar charts of phoneme frequencies."""
    # Convert to Pandas just for Plotly convenience
    df_proper = proper_counts.to_pandas()
    df_actual = actual_counts.to_pandas()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_proper['proper_phones'], y=df_proper['count'], name='Proper Intended Phonemes', marker_color='#1f77b4'))
    fig.add_trace(go.Bar(x=df_actual['phonetic_text'], y=df_actual['count'], name='Actual Child Phonemes', marker_color='#ff7f0e'))

    fig.update_layout(
        title="Phoneme Frequency Distribution",
        xaxis_title="Phoneme",
        yaxis_title="Count",
        barmode='group',
        template="plotly_white"
    )
    fig.show()

def analyze_alignments(df: pl.DataFrame):
    """
    Calculates PER, Insertion/Deletion/Substitution counts, 
    and generates the 1-to-1 mapping for the confusion matrix.
    """
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_hits = 0
    
    alignment_pairs = []

    print("Aligning strings and calculating Phoneme Error Rate (PER)...")
    
    for proper, actual in df.select(["proper_phones", "phonetic_text"]).iter_rows():
        # Strip spaces and stress marks so we only align base phonetic characters
        ref = proper.replace(" ", "").replace("ˈ", "").replace("ˌ", "")
        hyp = actual.replace(" ", "").replace("ˈ", "").replace("ˌ", "")
        
        if not ref or not hyp:
            continue
            
        # process_characters gives us character-level (phoneme-level) alignment
        result = jiwer.process_characters(ref, hyp)
        
        total_substitutions += result.substitutions
        total_deletions += result.deletions
        total_insertions += result.insertions
        total_hits += result.hits
        
        # Extract the exact alignment mapping
        for chunk in result.alignments[0]:
            if chunk.type in ['equal', 'substitute']:
                for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    r_char = ref[chunk.ref_start_idx + i]
                    h_char = hyp[chunk.hyp_start_idx + i]
                    alignment_pairs.append({
                        "proper_phone": r_char, 
                        "actual_phone": h_char, 
                        "type": chunk.type
                    })
            elif chunk.type == 'delete':
                for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                    r_char = ref[chunk.ref_start_idx + i]
                    alignment_pairs.append({
                        "proper_phone": r_char, 
                        "actual_phone": "[DEL]", 
                        "type": "delete"
                    })
            elif chunk.type == 'insert':
                for i in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                    h_char = hyp[chunk.hyp_start_idx + i]
                    alignment_pairs.append({
                        "proper_phone": "[INS]", 
                        "actual_phone": h_char, 
                        "type": "insert"
                    })

    # --- Print Insights ---
    total_reference_phones = total_hits + total_substitutions + total_deletions
    per = (total_substitutions + total_deletions + total_insertions) / total_reference_phones

    print("\n--- Alignment Insights ---")
    print(f"Total Reference Phonemes: {total_reference_phones:,}")
    print(f"Total Matches (Hits):     {total_hits:,}")
    print(f"Substitutions:            {total_substitutions:,}")
    print(f"Deletions:                {total_deletions:,}")
    print(f"Insertions:               {total_insertions:,}")
    print("-" * 26)
    print(f"Overall Phoneme Error Rate (PER): {per:.2%}\n")

    return pl.DataFrame(alignment_pairs)

def plot_interactive_confusion_matrix(pairs_df: pl.DataFrame):
    """Plots an interactive heatmap of phoneme substitutions, ignoring perfect matches."""
    
    # Filter out perfect matches so the errors stand out
    errors_only = pairs_df.filter(pl.col("type") != "equal")
    
    matrix_data = (
        errors_only.group_by(["proper_phone", "actual_phone"])
        .len(name="count")
    )
    
    # Pivot into a 2D matrix
    matrix_df = matrix_data.pivot(
        values="count", 
        index="proper_phone", 
        on="actual_phone"
    ).fill_null(0)

    # Convert to Pandas for Plotly
    pd_matrix = matrix_df.to_pandas().set_index("proper_phone")

    # Build the interactive heatmap
    fig = px.imshow(
        pd_matrix,
        labels=dict(x="What the Child Said", y="Proper Intended Phoneme", color="Count"),
        title="Interactive Phoneme Error Matrix (Substitutions, Deletions, Insertions)",
        color_continuous_scale="Viridis",
        aspect="auto"
    )

    fig.update_traces(
        hovertemplate="<b>Intended:</b> %{y}<br><b>Child Said:</b> %{x}<br><b>Count:</b> %{z}<extra></extra>"
    )

    fig.update_layout(
        width=1000, 
        height=900,
        xaxis_nticks=len(pd_matrix.columns),
        yaxis_nticks=len(pd_matrix.index)
    )

    fig.show()

def main():
    print("Loading Parquet file...")
    df = pl.read_parquet(PARQUET_FILE)
    
    # Drop rows missing transcripts
    df_clean = df.drop_nulls(subset=["proper_phones", "phonetic_text"])
    
    # 1. Phoneme Counts
    proper_counts = get_phoneme_counts(df_clean, "proper_phones")
    actual_counts = get_phoneme_counts(df_clean, "phonetic_text")
    
    print("\nTop 5 Proper Phonemes:")
    print(proper_counts.head(5))
    print("\nTop 5 Actual Phonemes:")
    print(actual_counts.head(5))
    
    # Plot the distributions
    plot_distributions(proper_counts, actual_counts)
    
    # 2. Alignment, PER, and Insights
    pairs_df = analyze_alignments(df_clean)
    
    # 3. Interactive Confusion Matrix
    plot_interactive_confusion_matrix(pairs_df)

if __name__ == "__main__":
    main()