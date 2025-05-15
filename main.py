from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import re
import traceback
from urllib.parse import urlparse
import string
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

def get_domain_specific_selectors(domain):
    """Return content selectors specific to known domains for better extraction"""
    selectors = {
        'medium.com': ['article', '.section-content'],
        'bbc.com': ['article', '.ssrcss-11r1m41-RichTextComponentWrapper'],
        'bbc.co.uk': ['article', '.ssrcss-11r1m41-RichTextComponentWrapper'],
        'nytimes.com': ['article', '.StoryBodyCompanionColumn'],
        'washingtonpost.com': ['article', '.article-body'],
        'theguardian.com': ['article', '.article-body-commercial-selector'],
        'cnn.com': ['article', '.article__content'],
        'reuters.com': ['article', '.ArticleBody__content___2gQno2'],
        'bloomberg.com': ['article', '.body-content'],
        'techcrunch.com': ['article', '.article-content'],
        'wired.com': ['article', '.article__body'],
        'arstechnica.com': ['article', '.article-content'],
        'forbes.com': ['article', '.article-body'],
        'firstpost.com': ['article', '.article-body', '.articleBody', '#_rcpb']
    }
    
    return selectors.get(domain, ['article', '.post-content', '.entry-content', '.content', '.post', '.article'])

def clean_text(text):
    """Clean and normalize text"""
    # Replace newlines and excessive spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any HTML tags that might remain
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and normalize
    text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')    
    return text.strip()

def extract_article_content(url):
    """Extract the main content from an article URL with improved accuracy"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch content: {str(e)}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove script, style elements, and comments
    for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
        element.decompose()
    
    # Get the domain for domain-specific extraction
    domain = urlparse(url).netloc.replace('www.', '')
    if 'medium.com' in domain:
        domain = 'medium.com'
    
    # Get title - try different approaches
    title = None
    title_element = soup.find('meta', property='og:title')
    if title_element and title_element.get('content'):
        title = title_element['content']
    if not title:
        title_element = soup.find('title')
        if title_element:
            title = title_element.text
    if not title and soup.h1:
        title = soup.h1.text
    
    # Clean the title
    if title:
        title = clean_text(title)
    else:
        title = "Article Summary"
    
    # Try to find the article content using domain-specific selectors
    selectors = get_domain_specific_selectors(domain)
    content_candidates = []
    
    # Try each selector
    for selector in selectors:
        if selector.startswith('.'):
            elements = soup.select(selector)
            for element in elements:
                content_candidates.append(element.get_text())
        else:
            elements = soup.find_all(selector)
            for element in elements:
                content_candidates.append(element.get_text())
    
    # If no content found through selectors, use paragraph extraction
    if not content_candidates:
        # Get all paragraphs with meaningful content
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            if len(p.get_text().strip()) > 50:  # Minimum length to filter out menu items, etc.
                content_candidates.append(p.get_text())
    
    # Clean and join the content
    content = ' '.join([clean_text(text) for text in content_candidates])
    
    # If we still don't have significant content, try a broader approach
    if len(content.split()) < 100:
        # Extract all text from the body
        body = soup.find('body')
        if body:
            # Remove navigation, header, footer sections that may exist
            for unwanted in body.find_all(['nav', 'header', 'footer', 'aside']):
                unwanted.decompose()
            
            # Get the remaining text
            content = clean_text(body.get_text())
    
    return title, content

def sentence_similarity(sent1, sent2, stopwords=None):
    """Calculate the cosine similarity between two sentences"""
    if stopwords is None:
        stopwords = set(stopwords.words('english'))
    
    sent1 = [w.lower() for w in word_tokenize(sent1) if w.lower() not in stopwords and w not in string.punctuation]
    sent2 = [w.lower() for w in word_tokenize(sent2) if w.lower() not in stopwords and w not in string.punctuation]
    
    # If either sentence is empty after removing stopwords, return 0
    if len(sent1) == 0 or len(sent2) == 0:
        return 0.0
    
    # Create word frequency dictionaries
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Build vectors
    for w in sent1:
        vector1[all_words.index(w)] += 1
    for w in sent2:
        vector2[all_words.index(w)] += 1
    
    # Calculate cosine similarity
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    """Create a similarity matrix for all sentences"""
    # Initialize similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:  # Same sentence
                similarity_matrix[i][j] = 1.0
            else:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
    
    return similarity_matrix

def textrank_summarize(text, num_sentences=5):
    """Generate a summary using TextRank algorithm for better coherence"""
    stop_words = set(stopwords.words('english'))
    
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Handle case when there are fewer sentences than requested
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Build the similarity matrix
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    
    # Apply PageRank algorithm
    sentence_similarity_graph = np.array(sentence_similarity_matrix)
    scores = np.array([sum(sentence_similarity_graph[i]) for i in range(len(sentences))])
    
    # Normalize scores
    scores = scores / sum(scores)
    
    # Get top-ranked sentence indices
    ranked_indexes = np.argsort(scores)[::-1]
    top_sentence_indices = ranked_indexes[:num_sentences]
    
    # Sort indices to maintain original order
    top_sentence_indices = sorted(top_sentence_indices)
    
    # Generate summary
    summary = [sentences[i] for i in top_sentence_indices]
    
    # Check if the summary has enough content
    if len(' '.join(summary).split()) < 50 and len(sentences) > num_sentences + 2:
        # If summary is too short, include more sentences
        additional_indices = ranked_indexes[num_sentences:num_sentences+2]
        top_sentence_indices = sorted(list(top_sentence_indices) + list(additional_indices))
        summary = [sentences[i] for i in top_sentence_indices]
    
    # Ensure summary is coherent
    return ' '.join(summary)

def identify_key_entities(text):
    """Identify key entities (people, organizations, topics) in the text"""
    # For simplicity, we'll use a frequency-based approach
    # In a production system, you might want to use a Named Entity Recognition model
    
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    
    # Filter out stopwords and punctuation
    filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation and len(word) > 1]
    
    # Get most common words
    word_freq = Counter(filtered_words)
    return [word for word, freq in word_freq.most_common(5)]

def adaptive_summarization(text, title=None, min_sentences=4, max_sentences=10):
    """Adaptively summarize text based on its length and complexity"""
    sentences = sent_tokenize(text)
    
    if len(sentences) <= min_sentences:
        return text
    
    # Calculate the appropriate number of sentences
    text_length = len(text.split())
    
    if text_length < 500:
        num_sentences = min_sentences
    elif text_length < 1000:
        num_sentences = min(5, len(sentences) // 2)
    elif text_length < 2000:
        num_sentences = min(6, len(sentences) // 3 + 2)
    else:
        num_sentences = min(max_sentences, len(sentences) // 4 + 3)
    
    # Use TextRank to get the base summary
    summary = textrank_summarize(text, num_sentences)
    
    # If title is provided, ensure summary includes relevant information
    if title:
        title_words = set(word_tokenize(title.lower())) - set(stopwords.words('english'))
        
        # Check if key words from title are in the summary
        summary_words = set(word_tokenize(summary.lower()))
        title_coverage = len(title_words.intersection(summary_words)) / max(1, len(title_words))
        
        # If title words aren't well represented, add a sentence that covers them
        if title_coverage < 0.5 and len(sentences) > num_sentences:
            # Find a sentence that best represents the title
            best_sentence = None
            best_score = -1
            
            for sentence in sentences:
                if sentence in summary:
                    continue
                    
                sentence_words = set(word_tokenize(sentence.lower()))
                score = len(title_words.intersection(sentence_words)) / max(1, len(sentence_words))
                
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
            
            # If we found a good sentence, add it to the summary
            if best_sentence and best_score > 0.2:
                summary = best_sentence + " " + summary
    
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        url = request.form['url']
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return jsonify({'error': 'Invalid URL format. URL must start with http:// or https://'})
        
        # Extract article content
        title, content = extract_article_content(url)
        
        # Check if we successfully extracted content
        if not content or len(content.split()) < 20:
            return jsonify({'error': 'Could not extract meaningful content from this URL'})
        
        # Generate adaptive summary
        summary = adaptive_summarization(content, title)
        
        # Break the summary into paragraphs for better readability
        sentences = sent_tokenize(summary)
        if len(sentences) <= 3:
            formatted_summary = summary
        else:
            # Group into roughly 3-4 sentences per paragraph
            paragraphs = []
            current_paragraph = []
            
            for sentence in sentences:
                current_paragraph.append(sentence)
                if len(current_paragraph) >= 3:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            # Add any remaining sentences
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            formatted_summary = '\n\n'.join(paragraphs)
        
        return jsonify({
            'title': title,
            'summary': formatted_summary,
            'url': url
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': f'Error processing article: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)