# scaling-waddle
Parallel text processing engine
                                                                                                                                                                                                             
🚀 Parallel Text Processing System

A Python-based parallel document analyzer that extracts text, breaks it into chunks, and processes all chunks simultaneously using multi-core CPUs.
This system detects keyword patterns, calculates frequency, and generates visual insights for large documents including PDFs, DOCX, PPTX, and TXT.

🔍 Problem Statement

Manual searching inside large documents (books, papers, research PDFs) is slow and inefficient.
This project enables users to search any keyword they choose and get results quickly with frequency count and chunk-wise distribution.

❗ Why This Project?

To speed up text search & processing

Reduce manual reading time

Increase accuracy & accountability

Automate keyword-based content mining

🧠 Thought Process

Clean text from file

Split into chunks (batches)

Assign chunks → CPU workers

Run keyword detection in parallel

Generate output frequency results

Produce visual insights & CSV export

🛠 How It Was Solved

Core Task	Approach

Text Extraction	PyMuPDF, DOCX, PPTX support

Cleaning	Regex preprocessing

Chunking	NLTK sentence tokenization

Pattern Matching	Regex rule engine

Parallel Execution	ProcessPoolExecutor

Result Export	CSV + SQLite database

Visualization	Matplotlib plots

📦 Features

✔ Supports PDF / DOCX / PPTX / TXT
✔ Multi-Core Parallel Execution
✔ Custom Keyword Input
✔ Predefined ML keyword rules
✔ Frequency Graphs & Insights
✔ output_rules.csv + rules.db generated

🧩 Keywords Detected Automatically
Category	Examples
**Neural Networks	deep network, feedforward, etc.
Layers	hidden layer, residual layer, pooling
Neurons	relu neuron, tanh neuron
Operations	backpropagation, gradient descent
Architectures	CNN, RNN, Transformers, LSTM
Training Terms	epoch, optimizer, batch size
Math Terms	matrix, eigenvalues, dot product
Probability	bayes, entropy, distribution
Evaluation	accuracy, loss, F1Score
**
⚙ Architecture / Flow
**Load Document → Extract Text → Preprocess → Chunk Text
          ↓                      ↓
   Parallel CPU Workers  ←  Regex Rule Engine
          ↓
   CSV + DB Output + Visualization Graphs**

📂 Output Files Generated
File	Description
output_rules.csv	Keyword counts per chunk
rules.db	SQLite storage of results
Graph Visuals	Match frequency charts


▶ Running the Program
1. Install Requirements
pip install PyMuPDF nltk python-docx python-pptx pandas matplotlib

2. Run Script
python rule_analyzer.py

3. Input Required
Enter file path: example.pdf  
Enter workers: 4  
Enter custom keywords: Matrix, Learning, Neural  

🔮 Future Enhancements

Web UI using Streamlit/Flask

Heatmap keyword distribution

Topic summarization using NLP

Auto-classification of document topic

📜 License

Open Source — free to modify & use.

⭐ If you like this project, consider giving a star 🌟
Parallel Text Processing — Fast. Smart. Scalable.
