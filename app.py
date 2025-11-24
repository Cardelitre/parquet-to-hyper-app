import streamlit as st
import pandas as pd
import numpy as np
from tableauhyperapi import HyperProcess, Telemetry, Connection, CreateMode, NOT_NULLABLE, NULLABLE, SqlType, TableDefinition, Inserter, TableName
import os
import time
import shutil
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import gc
import zipfile

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

def compress_file(input_path):
    """
    Compresses the file to .zip to save space and enable downloading larger datasets
    within Streamlit's memory limits. Returns path to zip file.
    """
    zip_path = str(input_path).replace('.hyper', '.zip')
    
    # Write to zip file in chunks to avoid memory spikes
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_path, arcname=os.path.basename(input_path))
        
    return zip_path

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
                for batch in pq_file.iter_batches(batch_size=BATCH_SIZE):
                    
                    if time.time() - start_time > TIMEOUT_SECONDS:
                        raise TimeoutError("Processing timeout exceeded (600 seconds).")
                    
                    df_chunk = batch.to_pandas()
                    df_chunk = clean_data(df_chunk)
                    
                    if table_def is None:
                        table_name = TableName("Extract", "Extract")
                        columns = []
                        for col_name, dtype in df_chunk.dtypes.items():
                            sql_type = get_hyper_type(dtype)
                            columns.append(TableDefinition.Column(name=str(col_name), type=sql_type, nullability=NULLABLE))
                        
                        table_def = TableDefinition(table_name=table_name, columns=columns)
                        connection.catalog.create_schema(schema=table_name.schema_name)
                        connection.catalog.create_table(table_definition=table_def)
                    
                    with Inserter(connection, table_def) as inserter:
                        rows = df_chunk.values.tolist()
                        inserter.add_rows(rows)
                        inserter.execute()
                    
                    rows_processed += len(df_chunk)
                    current_progress = min(rows_processed / total_rows, 1.0)
                    progress_bar.progress(current_progress)
                    status_text.text(f"Processing row {rows_processed} of {total_rows} total rows")
                    
                    # Memory Cleanup per chunk
                    del df_chunk
                    del rows
                    
                    # Aggressive PyArrow cleanup
                    if rows_processed % (BATCH_SIZE * 5) == 0:
                        pa.default_memory_pool().release_unused()
                        gc.collect()
        
        # Cleanup input file
        os.remove(parquet_path)
        
        # 5. Compress Result (New Step)
        status_text.text("Compressing output file...")
        zip_path = compress_file(hyper_path)
        
        # Delete the uncompressed hyper file to free disk/overhead
        os.remove(hyper_path)
        
        # Final Memory Cleanup
        pa.default_memory_pool().release_unused()
        gc.collect()
        
        return zip_path

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
    Convert your `.parquet` files to Tableau `.hyper` format.
    """)
    
    st.info("ℹ️ Numeric columns (int/float) will have NaNs and Inf replaced with 0. Mixed columns are treated as text.")

    # Initialize Session State
    if 'converted_file_path' not in st.session_state:
        st.session_state['converted_file_path'] = None

    # --- STATE 1: RESULT AVAILABLE (Download Mode) ---
    if st.session_state['converted_file_path'] and os.path.exists(st.session_state['converted_file_path']):
        
        st.success("✅ Conversion & Compression Completed!")
        st.markdown("Your file is ready. The upload form has been hidden to free up memory for the download.")
        
        file_path = st.session_state['converted_file_path']
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        st.write(f"**Output ZIP Size:** {file_size_mb:.2f} MB")
        
        # Final GC before download load
        gc.collect()
        
        with open(file_path, "rb") as file:
            st.download_button(
                label="⬇️ Download .zip File",
                data=file,
                file_name=os.path.basename(file_path),
                mime="application/zip"
            )
            
        st.markdown("---")
        if st.button("🔄 Convert Another File"):
            try:
                os.remove(file_path)
            except:
                pass
            st.session_state['converted_file_path'] = None
            st.rerun()

    # --- STATE 2: UPLOAD MODE ---
    else:
        uploaded_file = st.file_uploader("Choose a Parquet file", type="parquet")

        if uploaded_file is not None:
            # Check file size
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error("File size too large, max 1 GB")
                return

            st.subheader("File Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Filename:** {uploaded_file.name}")
            with col2:
                st.write(f"**Size:** {file_size_mb:.2f} MB")

            if st.button("Convert to Hyper"):
                with st.spinner("Initializing conversion process..."):
                    result_path = convert_parquet_to_hyper(uploaded_file)
                
                if result_path:
                    st.session_state['converted_file_path'] = str(result_path)
                    st.rerun()

if __name__ == "__main__":
    main()
