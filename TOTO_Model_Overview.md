# TOTO Model Overview

The `toto` directory contains the core components of a time series foundation model. The model's architecture processes data through a series of distinct steps, with a notable two-stage attention mechanism that handles both temporal and inter-series relationships.

---

## Data Flow Overview

The flow of data through the model is orchestrated by the **`TotoBackbone`** class  
[cite: `toto/toto_backbone.py`].  
A typical forward pass proceeds as follows:

### 1. Scaling
The input time series data is first normalized using a scaler  
[cite: `toto/toto_backbone.py`].  

- The model can use either `StdMeanScaler` or `CausalStdMeanScaler`.  
- These scalers transform the data to have zero mean and unit variance  
  [cite: `tests/test_scalers.py`].

---

### 2. Patching and Embedding
The scaled data is then fed into the **`PatchEmbedding`** layer  
[cite: `toto/toto_backbone.py`].

- The time series is divided into fixed-size chunks called *patches*  
  [cite: `toto/embedding.py`].  
- Patches are linearly projected into a higher-dimensional latent space  
  [cite: `toto/embedding.py`].  
- The patching process asserts that no single patch spans multiple identities  
  (e.g., different cities or products)  
  [cite: `toto/embedding.py`].

---

### 3. Transformer Processing (Core)
The patched and embedded data is passed to the **`TotoTransformer`**  
[cite: `toto/toto_backbone.py`].  
This is where the model’s main innovations lie  
[cite: `tests/test_transformer.py`].

The transformer operates in two sequential stages  
[cite: `tests/test_transformer.py`]:

#### a. Time-wise Attention
- Processed through **`TimeWiseMultiheadAttention`** layers  
  [cite: `toto/attention.py`, `toto/toto_transformer.py`].  
- **Causal attention** ensures outputs at each time step depend only on the current and past steps  
  [cite: `tests/test_attention.py`, `tests/test_transformer.py`].  
- Uses **Rotary Position Embeddings (RoPE)** to encode relative positional information  
  [cite: `toto/embedding.py`, `tests/test_embedding.py`, `toto/toto_transformer.py`].

#### b. Space-wise Attention
- Processed through **`SpaceWiseMultiheadAttention`** layers  
  [cite: `toto/attention.py`, `toto/toto_transformer.py`].  
- **Bidirectional attention** handles relationships across different time series within a batch  
  [cite: `tests/test_attention.py`].  
- Employs an **identity mask** to block attention across unrelated groups  
  [cite: `tests/test_attention.py`, `toto/attention.py`].  
- This separates inter-series modeling from standard transformer workflows.

---

### 4. Unembedding
After the transformer layers, the output is passed to an **`Unembed`** layer  
[cite: `toto/toto_backbone.py`].

- Applies a linear projection to map hidden states back to patch size  
  [cite: `toto/unembedding.py`].  
- Reshapes tensors to reconstruct the continuous time series representation  
  [cite: `toto/unembedding.py`].

---

### 5. Unscaling
Finally, the output is *unscaled* by reapplying the mean and standard deviation  
from the initial scaling step  
[cite: `toto/toto_backbone.py`].  
This restores the data to its original scale  
[cite: `toto/toto_backbone.py`].

---

## Unique Architectural Aspects

The **Toto model** differs from other time series approaches in several key ways:

### 🔹 Two-Stage Attention
- Sequential **time-wise → space-wise** processing  
  [cite: `toto/toto_transformer.py`].  
- First captures temporal dependencies, then inter-series relationships  
  [cite: `tests/test_attention.py`, `tests/test_transformer.py`].

### 🔹 Causal vs. Bidirectional Attention
- **Time-wise attention:** causal masking enforces autoregressive forecasting  
  [cite: `toto/attention.py`, `tests/test_attention.py`].  
- **Space-wise attention:** bidirectional masking with identity boundaries ensures correct inter-series handling  
  [cite: `toto/attention.py`, `tests/test_attention.py`, `tests/test_transformer.py`].

### 🔹 Root Mean Square Normalization (RMSNorm)
- Replaces common LayerNorm with **RMSNorm** inside each `TotoTransformerLayer`  
  [cite: `toto/toto_transformer.py`, `toto/normalise.py`].  
- RMSNorm normalizes tensors by their root mean square, providing a simpler and often faster alternative  
  [cite: `toto/normalise.py`].

---
