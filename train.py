import data
import chunking
import embeddings
import evaluation

def main():
    print("Starting full pipeline training...")
    
    print("\n--- Step 1: Data Processing & EDA ---")
    data.main()
    
    print("\n--- Step 2: Chunking ---")
    chunking.main()
    
    print("\n--- Step 3: Embeddings & FAISS Index ---")
    embeddings.main()
    
    print("\n--- Step 4: Evaluation ---")
    evaluation.main()
    
    print("\nPipeline execution complete! You can now run `python app.py`.")

if __name__ == '__main__':
    main()
