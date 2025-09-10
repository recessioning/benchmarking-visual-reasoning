## Benchmarking Visual Reasoning Capabilities of Large Language Models for Sentiment Detection

This project evaluates and compares the performance of different multimodal Large Language Models (LLMs) for classifying human facial sentiment. It provides a framework for testing models from Google, OpenAI, and local models via Ollama against a labeled dataset of facial expressions.

The primary goal is to analyze the accuracy, precision, and recall of these models in a zero-shot and few-shot setting, using various prompting strategies.

## Features

- **Multi-Provider Support**: Switch between different vision models by using the dedicated notebooks:

    - `classifier-google.ipynb`
    - `classifier-openai.ipynb`
    - `classifier-ollama.ipynb`

- **Prompt Engineering**: Test different system prompts, from simple instructions to a detailed prompt based on the Facial Action Coding System (F.A.C.S.).

- **Few-Shot Learning**: Includes a structure to provide few-shot examples to the model to improve classification accuracy.

- **Fail-safe Implementation**: Exponential backoff and retry logic for handling API errors (Note: for `classifier-google.ipynb` only).

- **Common Metrics Evaluation**: Automatically calculates and exports performance metrics, including accuracy, precision, recall, F1-score (macro, micro, and per-class), and a confusion matrix.

- **Outputs**: Saves raw prediction results to a .csv file and detailed evaluation metrics to a .json file for easy analysis and comparison.

## Project Structure

```
├── metrics/
├── results/
├── classifier-google.ipynb
├── classifier-openai.ipynb
├── classifier-ollama.ipynb
└── README.md
```

## Getting Started

Follow these steps to set up and run the project.

1. **Prerequisites**

- Python 3.8+

- An active internet connection (for Google and OpenAI models)

- Ollama installed and running with a multimodal (vision) model (for example, `ollama run gemma3:27b-it-q8`) for `classifier-ollama.ipynb`.

2. **Installation**

    1. **Clone the repository**:

    ```bash
    git clone https://github.com/realearn-people/benchmarking-visual-reasoning.git
    cd benchmarking-visual-reasoning
    ```

    2. **Install the required Python packages:**

    ```bash
    pip install python-dotenv scikit-learn google openai pandas ollama
    ```

3. **Configuration**

    1. **Set up the dataset:**
    Place your image dataset inside the `Dataset/Face/` directory (RAVDESS dataset was used), organized into subfolders named Positive, Negative, and Neutral according to the sentiment they represent.

    2. **Create an environment file:**
    Create a file named `.env` in the root of the project and add your API keys. The `classifier-ollama.ipynb` does not require an API key.

    ```
    # .env file
    GOOGLE_API_KEY="google-api-key"
    OPENAI_API_KEY="openai-api-key"
    ```

4. **Running the Notebooks**

    1. Open the notebook corresponding to the provider you want to test. For example, `classifier-google.ipynb`.

    2. Configure the parameters in the main execution block near the end of the notebook:
       
    ```bash
    # Select the model to use
    model = 'gemini-2.5-pro' # Or 'gpt-4o' for OpenAI, etc.

    # Set inference parameters
    temperature = 0
    top_p = 0.07

    # Choose the system prompt
    prompt = facs_prompt # Options: simple_prompt, system_prompt, facs_prompt

    # Set the number of runs per image
    num_runs = 3

    # Run the API request
    results = await api_request(
        dataset=dataset_test,
        client=client,
        model=model,
        temperature=temperature,
        top_p=top_p,
        prompt=prompt,
        # examples=examples_3, # Uncomment to enable few-shot
        num_runs=num_runs
    )
    ```

    3. Execute all cells in the notebook from top to bottom. The notebook will process the test dataset, print the progress, and save the output files.

## **Output**

After a successful run, the following files will be generated:

**Results** (`.csv`)

A CSV file will be created in the `results/` folder (e.g., `results/face_results_facs_gemini-2.5-pro_0_0.07.csv`). It contains the detailed prediction for each image.

**Metrics** (`.json`)

A JSON file containing the full evaluation report will be created in the `metrics/` folder (e.g., `metrics/face_metrics_facs_gemini-2.5-pro_0_0.07.json`).



