# StudyMate - Academic PDF Q&A System (Fixed Keyboard Interrupt)
import subprocess, sys

# Install packages
print("📦 Installing packages...")
for pkg in ["pymupdf", "sentence-transformers", "faiss-cpu", "transformers", "torch", "gradio", "accelerate"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import fitz, faiss, numpy as np, gradio as gr, torch, warnings, os, io, signal
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import files
warnings.filterwarnings('ignore')

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n⚠️  Interrupt received! Cleaning up...")
    shutdown_requested = True

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

print("🤖 Loading models...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.0-2b-instruct")
    llm_model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.0-2b-instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print("✅ Models loaded!\n")
except KeyboardInterrupt:
    print("\n❌ Model loading interrupted. Please restart the cell.")
    raise
except Exception as e:
    print(f"\n❌ Error loading models: {e}")
    raise

class StudyMate:
    def __init__(self):  # Fixed: was _init_ (single underscore)
        self.chunks, self.embeddings, self.index = [], None, None
        self.uploaded_files = []
        self.is_ready = False

    def process_pdf_bytes(self, pdf_bytes, filename):
        chunks = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for pg in range(len(doc)):
                # Check for interrupt
                if shutdown_requested:
                    print(f"\n⚠️  Processing interrupted at page {pg+1}")
                    doc.close()
                    return chunks

                text = doc[pg].get_text().split()
                for i in range(0, len(text), 450):
                    chunk = ' '.join(text[i:i+500])
                    if chunk.strip():
                        chunks.append({'text': chunk, 'page': pg+1, 'source': filename})
            doc.close()
        except KeyboardInterrupt:
            print(f"\n⚠️  PDF processing interrupted for {filename}")
            return chunks
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
        return chunks

    def upload_and_process(self):
        """Upload PDFs using Colab's native uploader (FAST!)"""
        global shutdown_requested
        shutdown_requested = False  # Reset flag

        print("\n" + "="*60)
        print("📤 UPLOADING PDFs - Click 'Choose Files' button below")
        print("="*60)

        try:
            uploaded = files.upload()
        except KeyboardInterrupt:
            print("\n⚠️  Upload cancelled by user")
            return "⚠️ Upload cancelled"

        if not uploaded:
            print("❌ No files uploaded")
            return "❌ No files uploaded"

        self.chunks = []
        self.uploaded_files = []

        print("\n🔄 Processing PDFs...")
        try:
            for filename, content in uploaded.items():
                if shutdown_requested:
                    print("\n⚠️  Processing stopped by user")
                    break

                if filename.lower().endswith('.pdf'):
                    chunks = self.process_pdf_bytes(io.BytesIO(content), filename)
                    if chunks:
                        self.chunks.extend(chunks)
                        self.uploaded_files.append(filename)
                        print(f"  ✓ {filename}: {len(chunks)} chunks")
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted")
            if self.chunks:
                print(f"✅ Partial progress saved: {len(self.chunks)} chunks from {len(self.uploaded_files)} file(s)")

        if not self.chunks:
            print("❌ No valid PDFs processed")
            return "❌ No valid PDFs processed"

        # Create embeddings
        print(f"\n🧠 Creating embeddings for {len(self.chunks)} chunks...")
        try:
            texts = [c['text'] for c in self.chunks]
            self.embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=32)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings.astype('float32'))
            self.is_ready = True
        except KeyboardInterrupt:
            print("\n⚠️  Embedding creation interrupted")
            self.is_ready = False
            return "⚠️ Processing incomplete - please try again"
        except Exception as e:
            print(f"\n❌ Error creating embeddings: {e}")
            return f"❌ Error: {e}"

        result = f"""
{'='*60}
✅ SUCCESS! PDFs PROCESSED
{'='*60}
📊 Files: {len(self.uploaded_files)}
📄 Chunks: {len(self.chunks)}
📚 Files: {', '.join(self.uploaded_files)}

🎯 NOW USE THE CHAT INTERFACE BELOW TO ASK QUESTIONS!
{'='*60}
"""
        print(result)
        return result

    def upload_and_process_gradio(self, files_list):
        """Upload PDFs from Gradio file component"""
        global shutdown_requested
        shutdown_requested = False

        if not files_list:
            return "❌ No files uploaded. Please select PDF files."

        self.chunks = []
        self.uploaded_files = []

        status_msg = "🔄 Processing PDFs...\n\n"

        try:
            for file_obj in files_list:
                if shutdown_requested:
                    status_msg += "\n⚠️ Processing stopped by user"
                    break

                filename = os.path.basename(file_obj.name)

                if filename.lower().endswith('.pdf'):
                    with open(file_obj.name, 'rb') as f:
                        pdf_bytes = f.read()

                    chunks = self.process_pdf_bytes(io.BytesIO(pdf_bytes), filename)
                    if chunks:
                        self.chunks.extend(chunks)
                        self.uploaded_files.append(filename)
                        status_msg += f"✓ {filename}: {len(chunks)} chunks\n"
        except Exception as e:
            status_msg += f"\n❌ Error: {str(e)}"
            return status_msg

        if not self.chunks:
            return "❌ No valid PDFs processed"

        # Create embeddings
        status_msg += f"\n🧠 Creating embeddings for {len(self.chunks)} chunks...\n"

        try:
            texts = [c['text'] for c in self.chunks]
            self.embeddings = embedding_model.encode(texts, show_progress_bar=False, batch_size=32)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings.astype('float32'))
            self.is_ready = True

            status_msg += f"\n{'='*60}\n"
            status_msg += "✅ SUCCESS! PDFs PROCESSED\n"
            status_msg += f"{'='*60}\n"
            status_msg += f"📊 Files: {len(self.uploaded_files)}\n"
            status_msg += f"📄 Chunks: {len(self.chunks)}\n"
            status_msg += f"📚 Files: {', '.join(self.uploaded_files)}\n\n"
            status_msg += "🎯 You can now ask questions in the chat below!"

        except Exception as e:
            self.is_ready = False
            status_msg += f"\n❌ Error creating embeddings: {e}"

        return status_msg

    def answer_question(self, question, history):
        """Answer question using RAG"""
        if not self.is_ready or not self.chunks:
            history = history or []
            history.append((question, "⚠️ Please upload PDFs first using the Upload tab!"))
            return history

        if not question or not question.strip():
            return history

        try:
            # Search
            q_emb = embedding_model.encode([question])
            _, indices = self.index.search(q_emb.astype('float32'), 3)
            chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

            if not chunks:
                history = history or []
                history.append((question, "❌ No relevant information found in the PDFs."))
                return history

            # Generate answer
            context = "\n\n".join([f"[Page {c['page']}] {c['text'][:400]}" for c in chunks])
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on context:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
            inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = llm_model.generate(**inputs, max_new_tokens=200, temperature=0.7,
                                         pad_token_id=tokenizer.eos_token_id)

            ans = tokenizer.decode(out[0], skip_special_tokens=True)
            ans = ans.split("Answer")[-1].strip().replace("Context:", "").replace("Question:", "")
            if ans.startswith(":"): ans = ans[1:].strip()

            # Add sources
            pages = sorted(set([c['page'] for c in chunks]))
            sources = set([c['source'] for c in chunks])
            ans += f"\n\n📚 *Sources:* {', '.join(sources)} - Pages {', '.join(map(str, pages))}"

            history = history or []
            history.append((question, ans))
            return history

        except KeyboardInterrupt:
            history = history or []
            history.append((question, "⚠️ Answer generation interrupted"))
            return history
        except Exception as e:
            history = history or []
            history.append((question, f"❌ Error: {str(e)}"))
            return history

    def get_status(self):
        """Get current status"""
        if self.is_ready:
            return f"✅ Ready! {len(self.uploaded_files)} file(s) loaded with {len(self.chunks)} chunks"
        return "⚠️ No PDFs loaded yet"

# Initialize engine
engine = StudyMate()

# Gradio Interface
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.header-box {
    background: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.upload-section {
    background: rgba(255,255,255,0.95);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 15px;
}
"""

with gr.Blocks(css=css, title="StudyMate") as demo:

    gr.HTML('''
    <div class="header-box">
        <h1>📚 StudyMate - AI Study Companion</h1>
        <p style="font-size: 18px; color: #666;">Powered by IBM Granite 3.0 2B & FAISS Semantic Search</p>
    </div>
    ''')

    with gr.Tabs() as tabs:
        # Upload Tab
        with gr.Tab("📤 Upload PDFs"):
            gr.HTML('''
            <div class="upload-section">
                <h2>📋 How to Upload</h2>
                <ol style="text-align: left; line-height: 1.8;">
                    <li>Click the "Upload Files" button below</li>
                    <li>Select one or more PDF files</li>
                    <li>Click "Process PDFs" button</li>
                    <li>Wait for processing to complete</li>
                    <li>Go to "Chat" tab to ask questions!</li>
                </ol>
            </div>
            ''')

            file_upload = gr.File(
                label="📁 Select PDF Files",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath"
            )

            upload_btn = gr.Button("🚀 Process PDFs", variant="primary", size="lg")

            upload_status = gr.Textbox(
                label="📊 Upload Status",
                value="⚠️ No PDFs uploaded yet. Please select files and click 'Process PDFs'",
                interactive=False,
                lines=10
            )

            gr.Markdown("""
            ---
            ### 💡 Alternative: Upload via Code Cell
            You can also upload PDFs by running this in a new code cell:
            ```python
            engine.upload_and_process()
            ```
            """)

        # Chat Tab
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    status_display = gr.Textbox(
                        label="📊 Current Status",
                        value="⚠️ No PDFs loaded. Please upload PDFs in the 'Upload' tab",
                        interactive=False,
                        lines=3
                    )

                    refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")

                    gr.Markdown("""
                    ### 💡 Tips
                    - Upload PDFs in the **Upload tab** first
                    - Ask clear, specific questions
                    - Questions are answered based on uploaded PDFs only
                    """)

                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Ask Questions About Your PDFs")

                    chatbot = gr.Chatbot(
                        label="StudyMate Assistant",
                        height=450,
                        avatar_images=(None, "🤖")
                    )

                    with gr.Row():
                        question_box = gr.Textbox(
                            label="Your Question",
                            placeholder="Type your question here...",
                            lines=2,
                            scale=4
                        )
                        ask_button = gr.Button("📨 Ask", variant="primary", scale=1)

                    with gr.Row():
                        clear_button = gr.Button("🗑 Clear Chat", variant="secondary")

                    gr.Examples(
                        examples=[
                            "What are the main topics covered in the document?",
                            "Summarize the key points",
                            "Explain the concept mentioned on page 5",
                            "What definitions are provided?",
                            "List the important formulas"
                        ],
                        inputs=question_box,
                        label="📝 Example Questions"
                    )

    # Event Handlers
    upload_btn.click(
        fn=engine.upload_and_process_gradio,
        inputs=[file_upload],
        outputs=[upload_status]
    )

    ask_button.click(
        fn=engine.answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot]
    ).then(lambda: "", outputs=[question_box])

    question_box.submit(
        fn=engine.answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot]
    ).then(lambda: "", outputs=[question_box])

    clear_button.click(lambda: None, outputs=[chatbot])

    refresh_btn.click(fn=engine.get_status, outputs=[status_display])

print("\n" + "="*70)
print("🎉 StudyMate is Ready!")
print("="*70)
print("\n📋 TWO WAYS TO UPLOAD PDFs:")
print("\n   METHOD 1 - Via Gradio Interface (Recommended):")
print("   1. Go to the 'Upload PDFs' tab in the interface above")
print("   2. Click 'Upload Files' and select your PDFs")
print("   3. Click 'Process PDFs' button")
print("   4. Wait for success message")
print("   5. Switch to 'Chat' tab to ask questions!")
print("\n   METHOD 2 - Via Code Cell:")
print("   1. Create a NEW CELL below this one")
print("   2. Type: engine.upload_and_process()")
print("   3. Run the cell (Shift+Enter)")
print("   4. Select your PDF files when prompted")
print("\n💡 TIP: Press Ctrl+C to safely interrupt operations")
print("="*70)
print("🚀 Launching interface...\n")

try:
    demo.launch(share=True, debug=False, inline=False)
except KeyboardInterrupt:
    print("\n⚠️  Gradio interface stopped by user")
except Exception as e:
    print(f"\n❌ Error launching interface: {e}")
