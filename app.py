from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_mail import Mail, Message
import os

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # innovatehive7@gmail.com
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # App Password
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')

# Initialize Flask-Mail
mail = Mail(app)

# Load models
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.6
)

# Directory for vector store
vector_store_dir = "vector_store"

# Check if vector store exists, if not create and save it
if not os.path.exists(os.path.join(vector_store_dir, "index.faiss")):
    from langchain.docstore.document import Document
    documents = [
        Document(page_content="This is a sample document about AI development."),
        Document(page_content="Another document about cloud computing."),
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(vector_store_dir)
else:
    vectorstore = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)

# Setup the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Form submission route
@app.route("/submit_form", methods=["POST"])
def submit_form():
    try:
        # Extract form data
        name = request.form.get("entry.2005620554")
        email = request.form.get("entry.1045781291")
        phone = request.form.get("entry.1166974658")
        service = request.form.get("entry.839337160")
        project_goals = request.form.get("entry.671065043")
        budget = request.form.get("entry.1223232703")
        message = request.form.get("entry.83890540")
        timestamp = request.form.get("timestamp")

        # Create email content
        email_body = f"""
        New Contact Form Submission
        -----------------------
        Name: {name}
        Email: {email}
        Phone: {phone or 'Not provided'}
        Service: {service}
        Project Goals: {project_goals or 'Not provided'}
        Budget: {budget or 'Not provided'}
        Message: {message}
        Timestamp: {timestamp}
        """

        # Create email message
        msg = Message(
            subject="New Contact Form Submission",
            recipients=['innovatehive7@gmail.com'],
            body=email_body
        )

        # Send email
        mail.send(msg)

        return jsonify({"message": "Form submitted successfully! We'll get back to you soon."}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to send email: {str(e)}"}), 500

# Route for all blogs page
@app.route("/all-blogs")
def all_blogs():
    return render_template("all_blogs.html")

# Additional routes for other pages
@app.route("/AI_Assistant_Development")
def ai_assistant_development():
    return render_template("AI_Assistant_Development.html")

@app.route("/ai1")
def ai1():
    return render_template("ai1.html")

@app.route("/ajinka")
def ajinka():
    return render_template("ajinka.html")

@app.route("/anuja")
def anuja():
    return render_template("anuja.html")

@app.route("/anurag")
def anurag():
    return render_template("anurag.html")

@app.route("/cloud")
def cloud():
    return render_template("cloud.html")

@app.route("/content")
def content():
    return render_template("content.html")

@app.route("/dhanashri")
def dhanashri():
    return render_template("dhanashri.html")

@app.route("/graphic")
def graphic():
    return render_template("graphic.html")

@app.route("/gurbani")
def gurbani():
    return render_template("gurbani.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/nuper")
def nuper():
    return render_template("nuper.html")

@app.route("/om_gws")
def om_gws():
    return render_template("om_gws.html")

@app.route("/om_shau")
def om_shau():
    return render_template("om_shau.html")

@app.route("/om")
def om():
    return render_template("om.html")

@app.route("/process")
def process():
    return render_template("process.html")

@app.route("/sahil")
def sahil():
    return render_template("sahil.html")

@app.route("/seo")
def seo():
    return render_template("seo.html")

@app.route("/sharv")
def sharv():
    return render_template("sharv.html")

@app.route("/siddhi")
def siddhi():
    return render_template("siddhi.html")

@app.route("/Video_Editing")
def video_editing():
    return render_template("Video_Editing.html")

@app.route("/website")
def website():
    return render_template("website.html")

# Chat API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    try:
        result = qa_chain.invoke({"query": user_input})
        return jsonify({"response": result["result"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)