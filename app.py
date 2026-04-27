from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from retrieval import load_index, retrieve_chunks
from generation import load_generator, generate_answer

app = Flask(__name__)

print("Initializing models and loading data...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index_loaded, chunks_loaded, _ = load_index('outputs/')
generator = load_generator()
print("App ready to serve requests.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    k = int(data.get('k', 5))
    
    if not question:
        return jsonify({'error': 'Please enter a valid question.'})
        
    try:
        retrieved = retrieve_chunks(question, model, index_loaded, chunks_loaded, k=k)
        answer, latency = generate_answer(question, retrieved, generator)
        
        return jsonify({
            'answer': answer,
            'latency': round(latency, 2),
            'sources': retrieved
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
