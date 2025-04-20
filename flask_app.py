from flask import Flask, request, render_template
from modal import App, Function

app = Flask(__name__)

# Initialize Modal app and function
modal_app = App("rag-pdf-processor")
query_function = Function.lookup("rag-pdf-processor", "query")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        question = request.form.get('question')
        if not question:
            return render_template('index.html', error="Please enter a question")
        
        try:
            # Call the query function directly
            result = query_function.remote(question)
            
            if "error" in result:
                return render_template('index.html', error=result["error"])
            
            return render_template('index.html',
                                question=question,
                                best_answer=result["best_answer"],
                                confidence=f"{result['confidence']:.2f}")
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)