# RAG Application with AWS Bedrock and LangChain

## Overview
Welcome to the **RAG App**, an advanced Retrieval-Augmented Generation (RAG) application leveraging AWS Bedrock, LangChain, and cutting-edge language models like Amazon Titan and Meta Llama. This project demonstrates a seamless pipeline for document ingestion, vector embedding, and answer generation, wrapped in a user-friendly Streamlit interface.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Key Technologies Used](#key-technologies-used)
- [Application Workflow](#application-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Snapshots](#snapshots)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This repository contains a powerful **Retrieval-Augmented Generation (RAG)** application built using **AWS Bedrock** and **LangChain**. The application is designed to process data, create vector embeddings, store them efficiently, and generate accurate answers to user queries using state-of-the-art language models like **Amazon Titan** and **Meta's Llama**.


## Features
- **Data Ingestion**: Load and process PDF files, splitting them into manageable chunks.
- **Vector Embeddings**: Utilize the **Amazon Titan Embedding Model** to generate high-quality vector embeddings of the document data.
- **Vector Storage**: Store and manage vector embeddings using the **FAISS (Facebook AI Similarity Search)** library and use it for efficient **similarity search** and clustering of vectors.
- **Language Models**: Generate responses using:
  - **Amazon Titan LLM**
  - **Meta Llama LLM** (via AWS Bedrock API)
- **Streamlit Interface**: An interactive UI built using **Streamlit** to make querying seamless and intuitive.

---

## Key Technologies Used

### Libraries and Frameworks
- **LangChain**: For orchestrating the RAG pipeline.
- **AWS Bedrock**: For embedding generation and LLM inference.
- **FAISS**: To efficiently store and retrieve vector embeddings.
- **Streamlit**: For creating a user-friendly web interface.
- **PyPDF**: Handles PDF file ingestion and splitting.
- **boto3**: AWS SDK for Python to interact with AWS services.

---

## Application Workflow

1. **Data Ingestion**:
   - PDF files are loaded and processed using the **PyPDF** library.
   - Documents are split into smaller chunks for embedding generation.

2. **Vector Embeddings**:
   - Create vector embeddings for text chunks using **Amazon Titan Embedding Model**.

3. **Vector Store**:
   - The FAISS library is employed to create and manage vector stores for efficient retrieval.

4. **Query Processing**:
   - Accept user queries via the **Streamlit** interface.
   - Retrieve relevant chunks from the vector store based on similarity.
   - Use **LangChain** to process the retrieved data and construct the final prompt.

5. **Answer Generation**:
   - Pass the constructed prompt to **Amazon Titan LLM** or **Meta Llama LLM** (via Bedrock API).
   - Display the generated answer in the Streamlit app.

---
## Snapshots
<img width="1440" alt="Screenshot 2025-01-10 at 8 09 51 PM" src="https://github.com/user-attachments/assets/e7ba61d2-ff14-4139-a7aa-bb072584fd9c" />
<img width="1440" alt="Screenshot 2025-01-10 at 8 09 59 PM" src="https://github.com/user-attachments/assets/c926fb90-4250-446e-a86c-812ea1802b0f" />
<img width="1440" alt="SS 1" src="https://github.com/user-attachments/assets/9e3175b9-f3e4-4744-9361-7a9eb03fda23" />
<img width="1440" alt="Screenshot 2025-01-10 at 8 17 09 PM" src="https://github.com/user-attachments/assets/2b0d00c6-313b-4913-9630-7be4b698866c" />
<img width="1440" alt="Screenshot 2025-01-10 at 8 17 28 PM" src="https://github.com/user-attachments/assets/cb356115-251a-476f-8af3-add751e2d33a" />
<img width="1440" alt="Screenshot 2025-01-10 at 8 17 56 PM" src="https://github.com/user-attachments/assets/785f34c9-292d-4f81-a9b4-91ea1acfe304" />
<img width="1440" alt="Screenshot 2025-01-10 at 8 18 17 PM" src="https://github.com/user-attachments/assets/1fedcfa9-47c4-4917-bb5f-58b3a4484cef" />
<img width="1440" alt="Screenshot 2025-01-10 at 8 18 52 PM" src="https://github.com/user-attachments/assets/c4b39abe-2749-4288-ab42-cb122dfacadd" />
<img width="1440" alt="Screenshot 2025-01-10 at 8 24 26 PM" src="https://github.com/user-attachments/assets/eca63d36-cc00-4d55-9478-5e28a95b372a" />
<img width="1440" alt="Screenshot 2025-01-10 at 8 24 40 PM" src="https://github.com/user-attachments/assets/7ad4c115-3157-4425-9278-ddc0cb291a53" />
<img width="1440" alt="Screenshot 2025-01-10 at 8 25 26 PM" src="https://github.com/user-attachments/assets/6c75d762-8e8c-4de0-b32d-a27ea4f291a3" />


---
## Installation

### Prerequisites
- Python 
- AWS account with Bedrock access

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure AWS credentials:
   ```bash
   aws configure
   ```
   Ensure you have the necessary permissions to access AWS Bedrock.
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## Usage
1. Place your PDF files in the `data/` folder.
2. Launch the application.
```
streamlit run app.py
```
3. Enter your query in the Streamlit interface.
4. Select the desired model (**Amazon Titan** or **Meta Llama**) for answer generation.
5. View the results in the app.

---


## Future Enhancements
- Add support for additional LLMs.
- Implement real-time feedback for query improvements.
- Enhance the Streamlit UI for better user experience.
- Optimize vector storage and retrieval for larger datasets.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [LangChain](https://langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
