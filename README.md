
# Generating Piano Music with Transformer

As my graduation project, I processed classical piano pieces using NLP and trained a generative Transformer-based model that continues a given piece of music in a musically coherent and original way. What it does is essentially to predict the next token (the next note/chord in my case). So, it tries to predict the ground-truth continuation to the given piece. However, it can't predict the next tokens with 100% accuracy, so as the autoregressive generation continues, the seed sequence becomes more synthetic, thus resulting in original generations.

This was an exploratory project to see if Transformer-based models can process music as well as they can process natural language and can a musical piece be represented in a symbolic way if it is some kind of a language. I was not expecting the model's generations to be highly appealing, especially given the trade-offs and small size of the data. However, the results are satisfying and the model is usually able to generate appealing sounds.

## Model Architecture & Training

### Model Overview

The core of this project is a **Transformer-based language model**, adapted to understand and generate symbolic piano music in a language-like format. The model treats each music token (e.g., `NOTE_C4`, `VALUE_Eighteenth`, `TIME_SHIFT_0.25`, `[CHORD_START]`) as a word, learning temporal structure and musical coherence through self-attention.

### Architecture

- **Embedding Layer**:  
  Maps each token to a dense vector. Positional encodings are optionally added to preserve the temporal order of tokens.

- **Transformer Blocks**:  
  Consist of stacked multi-head self-attention and feed-forward layers.

  - `N_layers`: 4  
  - `Heads`: 8  
  - `Hidden Size (d_model)`: 256   
  - `Dropout`: 0.25 

- **Output Layer**:  
  A final dense layer projects the hidden state to vocabulary size, followed by a `softmax` activation for next-token prediction.

### Hyperparameters

| Parameter         | Value         |
|------------------|---------------|
| Sequence Length   | 512 tokens    |
| Batch Size        | 32            |
| Epochs            | 10            |
| Learning Rate     | 1e-4          |
| Optimizer         | Adam          |
| Loss Function     | Sparse Categorical Crossentropy |
| Metric            | Accuracy      |

### Data and Preprocessing

- **Dataset**: Subset of MusicNet dataset (only the piano sonatas of Wolfgang Amadeus Mozart was used), tokenized to represent pitch, duration, and timing. The dataset comes with both MIDI and CSV files. MIDI files contains extra information in addition to the CSV files. Thus, several steps of preprocessing was applied to combine the files, in order to enrich the data.

- **Preprocessing**: MIDI files were used to label notes by which hand they were played. The notes that was played at the exact same time were combined to form chords. 

- **Tokenization**: All unique notes, note values and time shifts were considered a token along with special tokens such as [HAND_LEFT], [HAND_RIGHT], [CHORD_START], [CHORD_END], resulting in a vocabulary size of 223.

- **Sequence Strategy**: Fixed-length token sequences of size 512 were used. For the model to understand the music in a way similar to how humans do, I have experimented with various tokenization and sequence strategies. Here is a sample sequence:

  [HAND_LEFT] NOTE_D4 VALUE_Eighth TIME_SHIFT_0.500 [HAND_LEFT] NOTE_E3 VALUE_Eighth TIME_SHIFT_0.000 [HAND_RIGHT] [CHORD_START] NOTE_G#5 NOTE_G#4 VALUE_Eighth [CHORD_END] TIME_SHIFT_0.500

  [HAND_LEFT] and [HAND_RIGHT] tokens were used to help the model understand by which hand the notes were played.

  Creating a unique token for every chord in the dataset resulted in a massive vocabulary, which negatively effected model performance. Instead, I used [CHORD_START] and [CHORD_END] tokens to indicate that the notes between these tokens are played at the same time


### Post-processing and Testing

Since standard metrics like loss and accuracy don't fully capture musical quality, I evaluated generations subjectively by listening to samples. I wrote the required functions to post-process the generated token sequences and convert them into MIDI files. 10 sample generations created using random sequences from the test set can be found under `./generations`.

