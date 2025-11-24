import streamlit as st
import pandas as pd
import numpy as np
from tableauhyperapi import HyperProcess, Telemetry, Connection, CreateMode, NOT_NULLABLE, NULLABLE, SqlType, TableDefinition, Inserter, TableName
import os
import time
import shutil
from pathlib import Path
import pyarrow.parquet as pq
import gc  # Import Garbage Collector

# --- Configuration ---
st.set_page_config(
    page_title="Parquet to Hyper Converter",
    page_icon="🔄",
    layout="centered"
)

MAX_FILE_SIZE_MB = 1024  # 1 GB
TIMEOUT_SECONDS = 600
BATCH_SIZE = 10000  # Number of rows to process at a time

# --- Helper Functions ---

def get_hyper_type(dtype):
    """Maps Pandas dtypes to Tableau Hyper SqlTypes."""
    if pd.api.types.is_integer_dtype(dtype):
        return SqlType.big_int()
    elif pd.api.types.is_float_dtype(dtype):
        return SqlType.double()
    elif pd.api.types.is_bool_dtype(dtype):
        return SqlType.bool()
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return SqlType.timestamp()
    elif pd.api.types.is_timedelta64_dtype(dtype):
        return SqlType.text()
    else:
        return SqlType.text()

def clean_data(df):
    """
    Cleans numeric data: NaNs and Inf -> 0.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def convert_parquet_to_hyper(uploaded_file):
    """
    Main processing logic using Chunking/Streaming to save memory.
    """
    start_time = time.time()
    
    # Create a temporary directory
    temp_dir = Path("temp_conversion")
    temp_dir.mkdir(exist_ok=True)
    
    # Define file paths
    parquet_path = temp_dir / uploaded_file.name
    hyper_filename = f"{uploaded_file.name.replace('.parquet', '')}_{int(time.time())}.hyper"
    hyper_path = temp_dir / hyper_filename
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Save uploaded file to disk
        # We still need the file on disk to stream it, but this doesn't load it into RAM
        with open(parquet_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # 2. Initialize Parquet Stream
        status_text.text("Initializing Parquet stream...")
        try:
            pq_file = pq.ParquetFile(parquet_path)
            total_rows = pq_file.metadata.num_rows
        except Exception as e:
            st.error("Unable to read parquet file structure. File may be corrupted.")
            raise e

        # 3. Setup Hyper Process
        status_text.text("Initializing Hyper process...")
        
        with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
            with Connection(endpoint=hyper.endpoint, database=str(hyper_path), create_mode=CreateMode.CREATE_AND_REPLACE) as connection:
                
                table_def = None
                rows_processed = 0
                
                # 4. Iterate over batches (Chunking)
                # This prevents loading the whole file into RAM
                for batch in pq_file.iter_batches(batch_size=BATCH_SIZE):
                    
                    # Check Timeout
                    if time.time() - start_time > TIMEOUT_SECONDS:
                        raise TimeoutError("Processing timeout exceeded (600 seconds).")
                    
                    # Convert PyArrow batch to Pandas
                    df_chunk = batch.to_pandas()
                    
                    # Clean Data
                    df_chunk = clean_data(df_chunk)
                    
                    # Create Table Definition (Only on first chunk)
                    if table_def is None:
                        table_name = TableName("Extract", "Extract")
                        columns = []
                        for col_name, dtype in df_chunk.dtypes.items():
                            sql_type = get_hyper_type(dtype)
                            columns.append(TableDefinition.Column(name=str(col_name), type=sql_type, nullability=NULLABLE))
                        
                        table_def = TableDefinition(table_name=table_name, columns=columns)
                        connection.catalog.create_schema(schema=table_name.schema_name)
                        connection.catalog.create_table(table_definition=table_def)
                    
                    # Insert Data
                    with Inserter(connection, table_def) as inserter:
                        rows = df_chunk.values.tolist()
                        inserter.add_rows(rows)
                        inserter.execute()
                    
                    # Update Progress
                    rows_processed += len(df_chunk)
                    current_progress = min(rows_processed / total_rows, 1.0)
                    progress_bar.progress(current_progress)
                    status_text.text(f"Processing row {rows_processed} of {total_rows} total rows")
                    
                    # Explicitly free memory for this chunk
                    del df_chunk
                    del rows
                    # Force GC periodically (every ~100k rows) or just rely on the final sweep.
                    # Given the constraints, we rely on the final sweep to keep speed up.
        
        # Cleanup input file
        os.remove(parquet_path)
        
        # --- CRITICAL MEMORY FIX ---
        # Force garbage collection to release all memory from the processing step
        # before we attempt to load the file for download.
        gc.collect()
        
        return hyper_path

    except MemoryError:
        st.error("Insufficient memory. Even with chunking, a row group might be too large.")
        return None
    except TimeoutError as te:
        st.error(str(te))
        return None
    except Exception as e:
        if "Schema" in str(e):
             st.error("Schema processing failed. Check file structure.")
        else:
             st.error(f"An error occurred: {str(e)}")
        return None

# --- Main Application Flow ---

def main():
    st.title("Parquet to Tableau Hyper Converter")
    st.markdown("""
    Convert your `.parquet` files to Tableau `.hyper` format instantly.
    
    **Instructions:**
    1. Upload a valid `.parquet` file (Max 1GB).
    2. Wait for the processing to complete.
    3. Download your converted file.
    """)
    
    st.info("ℹ️ Numeric columns (int/float) will have NaNs and Inf replaced with 0. Mixed columns are treated as text.")

    # --- File Upload Section ---
    uploaded_file = st.file_uploader("Choose a Parquet file", type="parquet")

    if uploaded_file is not None:
        # Check file size (in bytes)
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error("File size too large, max 1 GB")
            return

        # Display File Stats
        st.subheader("File Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Filename:** {uploaded_file.name}")
        with col2:
            st.write(f"**Size:** {file_size_mb:.2f} MB")

        # --- Processing Section ---
        if st.button("Convert to Hyper"):
            # Clean previous runs in session state
            if 'converted_file_path' in st.session_state:
                del st.session_state['converted_file_path']
            
            with st.spinner("Initializing conversion process..."):
                result_path = convert_parquet_to_hyper(uploaded_file)
            
            if result_path:
                st.session_state['converted_file_path'] = str(result_path)
                st.success("Process completed successfully")
                st.balloons()

        # --- Download Section ---
        if 'converted_file_path' in st.session_state:
            file_path = st.session_state['converted_file_path']
            if os.path.exists(file_path):
                
                # --- CRITICAL MEMORY FIX ---
                # Force GC again before the download button loads the file
                gc.collect()
                
                try:
                    with open(file_path, "rb") as file:
                        st.download_button(
                            label="Download .hyper File",
                            data=file,
                            file_name=os.path.basename(file_path),
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"File too large to download in this environment: {str(e)}")

if __name__ == "__main__":
    main()
