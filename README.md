# TL;DR - Article Summarizer

A web application that generates concise, intelligent summaries of any article with a single click.

## 🚀 Features
- Instant Summaries: Transform long articles into bite-sized summaries in seconds
- Smart Extraction: Intelligently identifies and extracts the most important content
- Advanced Algorithms: Uses TextRank algorithm for coherent, accurate summaries
- URL-Based: Simply paste any article URL to get started
- Responsive Design: Works perfectly on desktop, tablet, and mobile devices
- Multi-Domain Support: Optimized for major news and blog platforms
- Advanced Error Handling: Gracefully handles invalid URLs and connection issues

## 🛠️ Technology Stack
- Backend: Python with Flask
- Frontend: HTML, CSS, JavaScript
- Text Processing: NLTK (Natural Language Toolkit)
- Content Extraction: BeautifulSoup, Requests
- Algorithms: TextRank (based on Google's PageRank)
- Containerization: Docker

## 🔧 Installation
<b>Using Docker (Recommended)</b>
```bash
# Clone the repository
git clone https://github.com/yourusername/tldr-summarizer.git
cd tldr-summarizer

# Build the Docker image
docker build -t tldr-app .

# Run the container
docker run -p 8080:8080 tldr-app
```

<b>Manual Installation</b>
```bash
# Clone the repository
git clone https://github.com/yourusername/tldr-summarizer.git
cd tldr-summarizer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the application
python main.py
```
<i>Visit http://localhost:8080 in your browser to use the application.</i>

## 📋 Usage
1. Open the application in your web browser
2. Paste any article URL into the input field
3. Click "Summarize"
4. Read the generated summary

## 🧠 How It Works
TL;DR uses a sophisticated pipeline to generate high-quality summaries:

1. URL Validation: Ensures the provided URL is valid and accessible
2. Content Extraction: Uses domain-specific extractors to identify the main article content
3. Text Preprocessing: Cleans and normalizes the extracted text
4. TextRank Algorithm: Builds a sentence similarity matrix and ranks sentences by importance
5. Adaptive Summarization: Selects the optimal number of sentences based on article length
6. Title Integration: Ensures the summary includes information relevant to the article title
7. Formatting: Arranges the summary into readable paragraphs

## 🤝 Contributing
Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

<b>Please ensure your code follows the project's style and includes appropriate tests.</b>

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements
- [NLTK](https://www.nltk.org/) for natural language processing tools
- [Flask](https://flask.palletsprojects.com/en/stable/) for the web framework
- [BeautifulSoup](https://pypi.org/project/beautifulsoup4/) for HTML parsing
- [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) algorithm for summarization
